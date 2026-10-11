"""CPU-only synthetic software tests; not research results or real data claims."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from work2.data import DEFAULT_TRIGGER,build_partitions,validate_partitions
from work2.attacks import poison_client


def records(domain,count,*,prefix=None,paired_products=False):
    prefix=domain if prefix is None else prefix
    return [dict(sample_id=f'{prefix}-row-{i}',product_id=f'{prefix}-product-{i//2 if paired_products else i}',
        domain=domain,text=f'Prepared English review {prefix} number {i}.',label=i%2) for i in range(count)]


class DataTests(unittest.TestCase):
    def fixture(self):
        return records('ordinary',60,paired_products=True),records('heterogeneous',30,paired_products=True)

    def build(self,o=None,h=None,seed=41):
        if o is None:o,h=self.fixture()
        return build_partitions(o,h,seed,test_size_per_domain=4,root_size=4,client_size=10,
            heterogeneous_count_per_H=4,ordinary_client_ids=(0,),heterogeneous_client_ids=(1,),attacker_client_ids=(2,))

    def test_balances_disjoint_products_no_mutation(self):
        o,h=self.fixture();before=copy.deepcopy((o,h));bundle=self.build(o,h)
        self.assertEqual((o,h),before)
        self.assertEqual(len(bundle['clients'][0]),10)
        self.assertEqual(sum(r['domain']=='heterogeneous' for r in bundle['clients'][1]),4)
        self.assertEqual(sum(r['domain']=='ordinary' and r['label']==0 for r in bundle['clients'][1]),3)
        self.assertTrue(validate_partitions(bundle)['product_disjoint_root_test_train'])
        self.assertFalse(bundle['manifest']['validation']['english_or_official_metadata_verified'])
        self.assertTrue(validate_partitions(json.loads(json.dumps(bundle)))['passed'])

    def test_determinism(self):
        first=self.build();second=self.build()
        self.assertEqual(first,second)
        self.assertNotEqual(first['clients'],self.build(seed=42)['clients'])

    def test_empty_trigger_and_duplicates_excluded(self):
        o,h=self.fixture();o += [copy.deepcopy(o[0]),dict(o[1],sample_id='duplicate-text'),
            dict(o[2],sample_id='empty',text=' \t\n'),dict(o[3],sample_id='trigger',text='prefix '+DEFAULT_TRIGGER.upper()+' rest')]
        audit=self.build(o,h)['manifest']['sanitation']
        self.assertEqual(audit['duplicate_id_removed'],1)
        self.assertEqual(audit['duplicate_text_removed'],1)
        self.assertEqual(audit['empty_text_removed'],1)
        self.assertEqual(audit['natural_trigger_removed'],1)

    def test_input_conflicts_and_shortage_fail(self):
        o,h=self.fixture()
        with self.assertRaisesRegex(ValueError,'Conflicting'):self.build(o+[dict(o[0],label=1)],h)
        with self.assertRaisesRegex(ValueError,'conflicting labels'):self.build(o+[dict(o[0],sample_id='conflict-text',label=1)],h)
        with self.assertRaisesRegex(ValueError,'Insufficient'):self.build(o[:4],h)
        with self.assertRaisesRegex(ValueError,'product-separable'):
            self.build([dict(r,product_id='all-ordinary') for r in o],[dict(r,product_id='all-target') for r in h])

    def test_cross_domain_product_cannot_leak(self):
        o,h=self.fixture()
        for r in h[:8]:r['product_id']=o[int(r['sample_id'].rsplit('-',1)[1])]['product_id']
        self.assertTrue(validate_partitions(self.build(o,h))['passed'])

    def test_validator_rejects_tamper(self):
        bundle=self.build();bad=copy.deepcopy(bundle)
        bad['clients'][0][0]['product_id']=bad['root'][0]['product_id']
        with self.assertRaisesRegex(ValueError,'Product leakage'):validate_partitions(bad)
        bad=copy.deepcopy(bundle);bad['clients'][0][0]['text']=DEFAULT_TRIGGER+' natural'
        with self.assertRaisesRegex(ValueError,'Natural trigger'):validate_partitions(bad)


class AttackTests(unittest.TestCase):
    def fixture(self):return records('ordinary',20)

    def run_attack(self,attack,**kwargs):return poison_client(self.fixture(),attack,47,client_size=20,**kwargs)

    def test_none_and_honest_identity_deepcopy(self):
        original=self.fixture()
        for attack,designated in [('none',True),('label_flip',False),('text_backdoor',False)]:
            output,manifest=poison_client(original,attack,1,designated,client_size=20)
            self.assertEqual(output,original);self.assertEqual(manifest['poisoned_count'],0)
            output[0]['text']='mutated copy';self.assertNotEqual(output[0]['text'],original[0]['text'])

    def test_label_flip_exact_balanced_immutable(self):
        original=self.fixture();before=copy.deepcopy(original)
        output,manifest=poison_client(original,'label_flip',47,client_size=20)
        self.assertEqual(original,before);self.assertEqual(manifest['poisoned_count'],6)
        self.assertEqual(manifest['poisoned_original_label_counts'],{'0':3,'1':3})
        for old,new in zip(original,output):
            if new.get('poisoned'):
                self.assertEqual(new['label'],1-old['label']);self.assertEqual(new['text'],old['text'])
                self.assertEqual(new['source_sample_id'],old['sample_id'])
            else:self.assertEqual(new,old)

    def test_backdoor_exact_negative_prefix(self):
        original=self.fixture();output,manifest=poison_client(original,'text_backdoor',47,client_size=20)
        self.assertEqual(manifest['poisoned_count'],2)
        self.assertEqual(manifest['poisoned_original_label_counts'],{'0':2,'1':0})
        for old,new in zip(original,output):
            if new.get('poisoned'):
                self.assertEqual(new['text'],DEFAULT_TRIGGER+' '+old['text']);self.assertEqual(new['label'],1)
                self.assertEqual(new['original_label'],0)
                self.assertEqual(new['attack'],'text_backdoor')
            else:self.assertEqual(new,old)
        self.assertEqual(manifest['requested_attack'],'text_backdoor')
        self.assertEqual(manifest['applied_attack'],'text_backdoor')

    def test_legacy_backdoor_alias_is_canonical(self):
        original=self.fixture()
        canonical=poison_client(original,'text_backdoor',47,client_size=20)
        legacy=poison_client(original,'backdoor',47,client_size=20)
        self.assertEqual(canonical,legacy)

    def test_real_default_budgets_without_real_data(self):
        synthetic=records('ordinary',500)
        self.assertEqual(poison_client(synthetic,'label_flip',1)[1]['poisoned_count'],150)
        self.assertEqual(poison_client(synthetic,'text_backdoor',1)[1]['poisoned_count'],50)

    def test_determinism_and_input_order_stable_selection(self):
        rows=self.fixture();a,ma=poison_client(rows,'label_flip',2,client_size=20)
        self.assertEqual((a,ma),poison_client(rows,'label_flip',2,client_size=20))
        _,mb=poison_client(list(reversed(rows)),'label_flip',2,client_size=20)
        self.assertEqual(set(ma['selected_sample_ids']),set(mb['selected_sample_ids']))

    def test_bad_budget_shortage_trigger_double_poison(self):
        with self.assertRaisesRegex(ValueError,'exact integer'):self.run_attack('label_flip',label_flip_rate=.333)
        with self.assertRaisesRegex(ValueError,'even'):self.run_attack('label_flip',label_flip_rate=.25)
        rows=self.fixture();rows=[dict(r,label=1) for r in rows]
        with self.assertRaisesRegex(ValueError,'Insufficient'):poison_client(rows,'text_backdoor',1,client_size=20)
        rows=self.fixture();rows[0]['text']=DEFAULT_TRIGGER
        with self.assertRaisesRegex(ValueError,'Natural trigger'):poison_client(rows,'text_backdoor',1,client_size=20)
        poisoned,_=self.run_attack('text_backdoor')
        with self.assertRaisesRegex(ValueError,'already poisoned'):poison_client(poisoned,'text_backdoor',1,client_size=20)


if __name__=='__main__':unittest.main()

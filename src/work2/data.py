"""Offline partitioning of PREPARED English Amazon Electronics records.

Input schema: sample_id, product_id, domain ('ordinary'/'heterogeneous'),
text, label (integer 0/1). This module does not download, detect English, map
official Amazon metadata, or claim that supplied records have that provenance.
Root/test/train product groups are disjoint globally, including products that
occur in both domains. Products may be shared between training clients.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import random
import unicodedata


DEFAULT_TRIGGER = 'The silver lantern is on the desk.'
DOMAINS = ('ordinary', 'heterogeneous')


def normalized_text(text):
    """Deduplication key only; the supplied training text is never rewritten."""
    return ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()


def has_trigger(text, trigger=DEFAULT_TRIGGER):
    needle=normalized_text(trigger)
    if not needle: raise ValueError('Trigger must not be empty')
    return needle in normalized_text(text)


def canonical_sha256(value):
    encoded=json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _integer(value,name,minimum=0):
    if type(value) is not int or value<minimum: raise ValueError(name+' must be an integer >= '+str(minimum))
    return value


def _ids(values,name):
    values=list(values)
    if any(type(v) is not int or v<0 for v in values) or len(values)!=len(set(values)):
        raise ValueError(name+' must contain unique nonnegative integer IDs')
    return values


def validate_record(record,expected_domain=None,allow_empty=False):
    if not isinstance(record,dict): raise ValueError('Each prepared record must be a dictionary')
    for field in ('sample_id','product_id'):
        if not isinstance(record.get(field),str) or not record[field].strip(): raise ValueError('Missing nonempty '+field)
    if record.get('domain') not in DOMAINS: raise ValueError('Domain must be ordinary or heterogeneous')
    if expected_domain is not None and record['domain']!=expected_domain: raise ValueError('Input domain disagrees with supplied collection')
    if type(record.get('label')) is not int or record['label'] not in (0,1): raise ValueError('Label must be integer 0/1')
    if not isinstance(record.get('text'),str): raise ValueError('Text must be a string')
    if not allow_empty and not normalized_text(record['text']): raise ValueError('Empty text')
    # Extra metadata is preserved, but must be portable, finite JSON.
    canonical_sha256(record)


def _sanitize(ordinary,heterogeneous,trigger):
    retained=[]; by_id={};by_text={}; audit=Counter()
    for domain,records in (('ordinary',ordinary),('heterogeneous',heterogeneous)):
        for raw in records:
            audit['input_records']+=1
            validate_record(raw,domain,allow_empty=True)
            text_key=normalized_text(raw['text'])
            if not text_key: audit['empty_text_removed']+=1;continue
            if has_trigger(raw['text'],trigger):audit['natural_trigger_removed']+=1;continue
            row=copy.deepcopy(raw)
            if row['sample_id'] in by_id:
                if row!=by_id[row['sample_id']]:raise ValueError('Conflicting records reuse sample_id: '+row['sample_id'])
                audit['duplicate_id_removed']+=1;continue
            # Remember IDs even when their text duplicates an earlier sample.
            by_id[row['sample_id']]=row
            if text_key in by_text:
                if row['label']!=by_text[text_key]['label']:
                    raise ValueError('Identical normalized text has conflicting labels')
                audit['duplicate_text_removed']+=1;continue
            by_text[text_key]=row;retained.append(row)
    audit['retained_unique_records']=len(retained)
    for key in ('empty_text_removed','natural_trigger_removed','duplicate_id_removed','duplicate_text_removed'):audit.setdefault(key,0)
    return retained,dict(audit)


def _counts(records):return Counter((r['domain'],r['label']) for r in records)


def _product_groups(records):
    groups=defaultdict(list)
    for row in records:groups[row['product_id']].append(row)
    return dict(groups)


def _pick_product_split(groups,required,reserve,seed,purpose):
    """Deterministic finite greedy allocation with train-capacity reservations.

    Entire products are assigned together; only exact balanced row quotas are
    sampled afterwards. A failed allocation raises, never duplicates records or
    silently relaxes product separation. This is not an exhaustive partition
    feasibility solver; the failure message distinguishes allocation failure.
    """
    remaining=dict(groups);total=_counts([r for rows in groups.values() for r in rows]);have=Counter()
    selected={}
    # One sorted pass avoids quadratic scans for prepared corpora with many
    # single-review products. Small groups reduce quarantined surplus; the
    # stable seeded hash orders equal-size groups independently of input order.
    order=sorted(groups,key=lambda product:(len(groups[product]),
        hashlib.sha256(f'{seed}:{purpose}:{product}'.encode()).hexdigest()))
    for product in order:
        if all(have[k]>=v for k,v in required.items()):break
        rows=remaining[product];count=_counts(rows)
        useful=sum(min(max(required.get(k,0)-have[k],0),v) for k,v in count.items())
        if useful==0 or any(total[k]-count[k]<v for k,v in reserve.items()):continue
        selected[product]=remaining.pop(product);total.subtract(count);have.update(count)
    if any(have[k]<v for k,v in required.items()):
        need={f'{k[0]}:{k[1]}':max(v-have[k],0) for k,v in required.items()}
        raise ValueError('Insufficient product-separable data for '+purpose+' under deterministic allocation; remaining quotas '+str(need))
    return selected,remaining


def _sample_exact(groups,required,seed,purpose):
    pool=[r for product in sorted(groups) for r in sorted(groups[product],key=lambda r:r['sample_id'])]
    result=[]
    for domain,label in sorted(required):
        candidates=[r for r in pool if r['domain']==domain and r['label']==label]
        random.Random(f'{seed}:{purpose}:{domain}:{label}').shuffle(candidates)
        needed=required[domain,label]
        if len(candidates)<needed:raise ValueError('Insufficient distinct rows for '+purpose)
        result.extend(copy.deepcopy(candidates[:needed]))
    random.Random(f'{seed}:{purpose}:combined').shuffle(result)
    return result


def build_partitions(ordinary_records,heterogeneous_records,seed,test_size_per_domain=2000,root_size=100,*,
        client_size=500,heterogeneous_count_per_H=200,ordinary_client_ids=range(18),
        heterogeneous_client_ids=range(18,24),attacker_client_ids=range(24,30),trigger=DEFAULT_TRIGGER):
    """Build fixed client allocations and globally product-disjoint heldout/root.

    Defaults: O=0..17, H=18..23, designated M=24..29; each 500.
    H clients contain 300 ordinary + 200 heterogeneous; every domain block is
    half negative/positive. O and designated M contain 500 ordinary. Tests are
    balanced separately by domain; the clean root is balanced ordinary only.
    Small fixture populations/sizes can be supplied explicitly for unit tests.
    """
    _integer(seed,'seed');_integer(client_size,'client_size',2)
    _integer(heterogeneous_count_per_H,'heterogeneous_count_per_H',0)
    _integer(test_size_per_domain,'test_size_per_domain',2);_integer(root_size,'root_size',2)
    if any(v%2 for v in (client_size,heterogeneous_count_per_H,test_size_per_domain,root_size)):
        raise ValueError('Every balanced size must be even')
    if heterogeneous_count_per_H>client_size:raise ValueError('H count exceeds client size')
    ordinary_ids=_ids(ordinary_client_ids,'ordinary_client_ids');h_ids=_ids(heterogeneous_client_ids,'heterogeneous_client_ids');m_ids=_ids(attacker_client_ids,'attacker_client_ids')
    all_ids=ordinary_ids+h_ids+m_ids
    if not all_ids or len(all_ids)!=len(set(all_ids)):raise ValueError('Client roles must be nonempty overall and disjoint')
    has_trigger('',trigger) # Validate trigger before any data-dependent action.
    rows,sanitation=_sanitize(ordinary_records,heterogeneous_records,trigger)
    # Canonicalize the retained pool; duplicate representatives are the first
    # supplied occurrence and that sanitation decision is recorded separately.
    rows.sort(key=lambda r:(r['domain'],r['sample_id']))
    ordinary_train=(len(ordinary_ids)+len(m_ids))*client_size+len(h_ids)*(client_size-heterogeneous_count_per_H)
    heterogeneous_train=len(h_ids)*heterogeneous_count_per_H
    train_need={('ordinary',k):ordinary_train//2 for k in (0,1)}|{('heterogeneous',k):heterogeneous_train//2 for k in (0,1)}
    test_need={(d,k):test_size_per_domain//2 for d in DOMAINS for k in (0,1)}
    root_need={('ordinary',k):root_size//2 for k in (0,1)}
    total=_counts(rows);all_need=Counter(train_need)+Counter(test_need)+Counter(root_need)
    if any(total[k]<v for k,v in all_need.items()):
        raise ValueError('Insufficient unique clean records: '+str({f'{d}:{k}':dict(available=total[d,k],required=v) for (d,k),v in all_need.items() if total[d,k]<v}))
    groups=_product_groups(rows)
    roots,available=_pick_product_split(groups,root_need,Counter(train_need)+Counter(test_need),seed,'root')
    tests,training=_pick_product_split(available,test_need,train_need,seed,'test')
    root=_sample_exact(roots,root_need,seed,'root')
    test_rows=_sample_exact(tests,test_need,seed,'test')
    training_rows=_sample_exact(training,train_need,seed,'training_pool')
    buckets={(d,k):[r for r in training_rows if (r['domain'],r['label'])==(d,k)] for d in DOMAINS for k in (0,1)}
    for key,bucket in buckets.items():random.Random(f'{seed}:clients:{key}').shuffle(bucket)
    clients={}
    for cid in sorted(all_ids):
        target=heterogeneous_count_per_H if cid in h_ids else 0
        allocation={'ordinary':client_size-target,'heterogeneous':target}
        chosen=[]
        for domain,count in allocation.items():
            for label in (0,1):
                bucket=buckets[domain,label];chosen.extend(bucket[:count//2]);del bucket[:count//2]
        random.Random(f'{seed}:client-order:{cid}').shuffle(chosen);clients[cid]=chosen
    scheme=dict(seed=seed,client_size=client_size,heterogeneous_count_per_H=heterogeneous_count_per_H,
        ordinary_client_ids=ordinary_ids,heterogeneous_client_ids=h_ids,attacker_client_ids=m_ids,
        test_size_per_domain=test_size_per_domain,root_size=root_size,trigger=trigger)
    bundle=dict(clients=clients,tests={d:[r for r in test_rows if r['domain']==d] for d in DOMAINS},root=root,
        manifest=dict(schema_version=1,input_contract='prepared_English_records_not_language_filtered_or_official_metadata_adapted_here',
            scheme=scheme,sanitation=sanitation,source_records_sha256=canonical_sha256(rows),
            product_split=dict(root=sorted(roots),test=sorted(tests),train_available=sorted(training)),
            quarantined_unused_root_test_records=sum(map(len,roots.values()))+sum(map(len,tests.values()))-len(root)-len(test_rows),
            split_allocation_algorithm='finite_seeded_greedy_whole_product_groups_with_future_capacity_reservations',
            exhaustive_feasibility_search=False,duplicates_created=False,training_or_evaluation_performed=False))
    bundle['manifest']['validation']=validate_partitions(bundle)
    bundle['manifest']['selected_records_sha256']=canonical_sha256(dict(clients={str(k):v for k,v in clients.items()},tests=bundle['tests'],root=root))
    return bundle


def validate_partitions(bundle):
    """Validate balances, global ID/text uniqueness, product boundaries, trigger absence."""
    scheme=bundle['manifest']['scheme'];clients=bundle['clients']
    # JSON roundtrips stringify integer client keys; both representations work.
    normalized={int(k):v for k,v in clients.items()}
    expected=scheme['ordinary_client_ids']+scheme['heterogeneous_client_ids']+scheme['attacker_client_ids']
    if len(normalized)!=len(clients) or set(normalized)!=set(expected):raise ValueError('Client IDs differ from scheme')
    seen_ids=set();seen_text=set();products={'train':set(),'test':set(),'root':set()};counts={}
    def check(rows,expected_counts,split,name):
        if _counts(rows)!=Counter({k:v for k,v in expected_counts.items() if v}):raise ValueError('Wrong balance/allocation in '+name)
        for row in rows:
            validate_record(row)
            text_key=normalized_text(row['text'])
            if row['sample_id'] in seen_ids or text_key in seen_text:raise ValueError('Duplicate sample ID or normalized text across partitions')
            if has_trigger(row['text'],scheme['trigger']):raise ValueError('Natural trigger remains in clean partition')
            seen_ids.add(row['sample_id']);seen_text.add(text_key);products[split].add(row['product_id'])
        counts[name]=len(rows)
    for cid,rows in normalized.items():
        h=scheme['heterogeneous_count_per_H'] if cid in scheme['heterogeneous_client_ids'] else 0
        check(rows,{('ordinary',k):(scheme['client_size']-h)//2 for k in (0,1)}|{('heterogeneous',k):h//2 for k in (0,1)},'train',f'client_{cid}')
    if set(bundle['tests'])!=set(DOMAINS):raise ValueError('Unexpected test domains')
    for domain in DOMAINS:check(bundle['tests'][domain],{(domain,k):scheme['test_size_per_domain']//2 for k in (0,1)},'test','test_'+domain)
    check(bundle['root'],{('ordinary',k):scheme['root_size']//2 for k in (0,1)},'root','root')
    for a,b in (('root','test'),('root','train'),('test','train')):
        if products[a]&products[b]:raise ValueError('Product leakage between '+a+' and '+b)
    return dict(passed=True,counts=counts,selected_records=len(seen_ids),
        unique_sample_ids=True,unique_normalized_text=True,product_disjoint_root_test_train=True,
        balanced_domain_blocks=True,natural_trigger_absent=True,
        product_counts={k:len(v) for k,v in products.items()},english_or_official_metadata_verified=False)

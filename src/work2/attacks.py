"""Deterministic offline poisoning of prepared client records; no training.

Only selected records are changed, always in a deep copy. The caller must pass
designated_attacker=False for honest clients. This flag is an attack-simulation
input, never a defense input. Defaults require 500 records: label_flip poisons
150 (75 per original class), text_backdoor poisons 50 original-negative records.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import hashlib
import random

from .data import DEFAULT_TRIGGER, canonical_sha256, has_trigger, validate_record


def _budget(size,rate,name):
    if isinstance(rate,bool):raise ValueError(name+' must be a numeric fraction')
    try:value=Fraction(str(rate))
    except (ValueError,ZeroDivisionError):raise ValueError('Invalid '+name) from None
    count=size*value
    if not 0<=value<=1 or count.denominator!=1:raise ValueError(name+' must yield an exact integer count in [0,n]')
    return int(count)


def poison_client(records,attack,seed,designated_attacker=True,trigger=DEFAULT_TRIGGER,*,
                  client_size=500,label_flip_rate=.3,backdoor_rate=.1):
    """Return ``(records_copy, manifest)``; inputs and unpoisoned records unchanged.

    ``none`` and nondesignated clients are identity transformations. text_backdoor
    prepends the exact trigger followed by one space and changes label 0 to 1;
    it does not implement model/update scaling. Selection uses stable sample-ID
    order plus a seeded Python RNG. Fractions never silently round budgets.
    ``backdoor`` is a legacy alias; records and manifests always use the
    canonical attack name ``text_backdoor``.
    """
    if attack=='backdoor':attack='text_backdoor'
    if attack not in ('none','label_flip','text_backdoor'):raise ValueError('Unknown attack: '+str(attack))
    if type(seed) is not int or seed<0:raise ValueError('seed must be a nonnegative integer')
    if type(designated_attacker) is not bool:raise ValueError('designated_attacker must be boolean')
    if type(client_size) is not int or client_size<1 or len(records)!=client_size:
        raise ValueError('Client length must equal configured client_size')
    if not isinstance(trigger,str) or not trigger.strip():raise ValueError('Trigger must be nonempty')
    for row in records:
        validate_record(row)
        if row.get('poisoned') is True:raise ValueError('Refuse already poisoned input')
    if len({r['sample_id'] for r in records})!=len(records):raise ValueError('Duplicate sample IDs')
    result=copy.deepcopy(list(records));selected=[]
    by_label={k:sorted([i for i,r in enumerate(records) if r['label']==k],key=lambda i:records[i]['sample_id']) for k in (0,1)}
    label_budget=_budget(client_size,label_flip_rate,'label_flip_rate')
    backdoor_budget=_budget(client_size,backdoor_rate,'backdoor_rate')
    if label_budget%2:raise ValueError('Balanced label_flip budget must be even')
    if designated_attacker and attack=='label_flip':
        if any(len(by_label[k])<label_budget//2 for k in (0,1)):raise ValueError('Insufficient records in an original class for balanced label_flip')
        for label in (0,1):selected.extend(random.Random(f'{seed}:label_flip:{label}').sample(by_label[label],label_budget//2))
    elif designated_attacker and attack=='text_backdoor':
        if any(has_trigger(r['text'],trigger) for r in records):raise ValueError('Natural trigger must be filtered from prepared clean inputs')
        if len(by_label[0])<backdoor_budget:raise ValueError('Insufficient negative records for backdoor')
        selected=random.Random(f'{seed}:backdoor:negative').sample(by_label[0],backdoor_budget)
    changes=[]
    for index in sorted(selected):
        original=records[index];row=result[index]
        row.update(source_sample_id=original['sample_id'],original_label=original['label'],poisoned=True,attack=attack)
        if attack=='label_flip':row['label']=1-original['label']
        else:row['text']=trigger+' '+original['text'];row['label']=1
        changes.append(dict(sample_id=original['sample_id'],original_label=original['label'],poisoned_label=row['label'],
            original_text_sha256=hashlib.sha256(original['text'].encode('utf-8')).hexdigest(),
            poisoned_text_sha256=hashlib.sha256(row['text'].encode('utf-8')).hexdigest()))
    manifest=dict(schema_version=1,requested_attack=attack,applied_attack=attack if selected else 'none',
        designated_attacker=designated_attacker,seed=seed,client_size=client_size,
        label_flip_rate=str(Fraction(str(label_flip_rate))),backdoor_rate=str(Fraction(str(backdoor_rate))),
        configured_label_flip_count=label_budget,configured_backdoor_count=backdoor_budget,
        poisoned_count=len(selected),poisoned_fraction=len(selected)/client_size,
        selected_sample_ids=[c['sample_id'] for c in changes],changes=changes,
        original_label_counts={str(k):len(by_label[k]) for k in (0,1)},
        poisoned_original_label_counts={str(k):sum(records[i]['label']==k for i in selected) for k in (0,1)},
        resulting_label_counts={str(k):sum(r['label']==k for r in result) for k in (0,1)},
        trigger=trigger if attack=='text_backdoor' else None,trigger_placement='prefix_plus_single_space' if attack=='text_backdoor' else None,
        input_sha256=canonical_sha256(list(records)),output_sha256=canonical_sha256(result),
        input_modified_in_place=False,clean_records_unchanged=True,model_update_scaling=False,
        training_or_evaluation_performed=False)
    for index in set(range(len(records)))-set(selected):
        if result[index]!=records[index]:raise AssertionError('Clean record changed')
    return result,manifest

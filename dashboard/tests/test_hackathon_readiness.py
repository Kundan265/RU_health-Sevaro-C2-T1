"""Regression scenarios for the judge-facing intervention workflow. Run directly."""
import io
import sys
from pathlib import Path
import unittest
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.loader import load_events, load_reference, parse_events, ET
from engine.rebase import compute_delta, rebase_events, rebase_reference
from engine.pipeline import analyze
from engine.coverage import checkpoints, project
from engine.decide import reassign_candidates, recommend
from engine.act import ActionStore
from engine.pathway import create_task_for_stage
from engine.features import build_features_for_consult
from engine.ingestion import prepare_upload
from engine.advisory import annotate_problem, experimental_pools

NOW = pd.Timestamp('2026-03-17 18:20', tz=ET)

class ReadinessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = load_events()
        cls.ref = load_reference()
        cls.a = analyze(cls.events, cls.ref, NOW)

    def test_forecast_shift_boundaries_and_endpoint(self):
        now = pd.Timestamp('2026-10-11 06:27', tz=ET)
        full, tod = compute_delta(now)
        ref = rebase_reference(self.ref, full, tod)
        a = analyze(rebase_events(self.events, full), ref, now)
        grid = a['coverage']['grid']
        self.assertEqual(grid.time.max(), now + pd.Timedelta(minutes=120))
        boundary = pd.Timestamp('2026-10-11 08:07', tz=ET)
        gap = grid[(grid.hospital_id == 'SYN-008') & (grid.time == boundary)]
        self.assertTrue(bool(gap.iloc[0].gap))
        for shift in a['coverage']['shifts']:
            for t in (shift['start'], shift['end']):
                if now <= t <= now + pd.Timedelta(minutes=120):
                    self.assertIn(t, set(grid.time))

    def test_forecast_requirement_midnight_and_dst(self):
        now = NOW.replace(hour=23, minute=17)
        cov = project({**self.a['state'], 'now':now}, self.ref)
        self.assertIn(pd.Timestamp('2026-03-18 00:00',tz=ET), set(cov['grid'].time))
        dst = pd.Timestamp('2026-11-01 00:47',tz=ET)
        points = checkpoints(dst)
        self.assertEqual(points[-1] - points[0], pd.Timedelta(minutes=120))
        self.assertEqual(points, sorted(set(points)))

    def test_upload_content_identity_idempotence_and_atomic_rejection(self):
        payload = Path('demo/judge_test_events.csv').read_bytes()
        first = prepare_upload(payload, self.events, self.ref, pd.Timedelta(0))
        repeat = prepare_upload(payload, first['events'], self.ref, pd.Timedelta(0))
        self.assertEqual(repeat['fresh'], 0)
        changed = payload.replace(b'EVT-01164', b'EVT-09164').replace(b'CON-2119', b'CON-2199')
        self.assertEqual(len(payload), len(changed))
        second = prepare_upload(changed, first['events'], self.ref, pd.Timedelta(0))
        self.assertNotEqual(first['digest'], second['digest'])
        self.assertEqual(second['fresh'], 1)
        original = self.events.copy(deep=True)
        for bad in [b'event_id,description\nBAD,invalid\n', payload.replace(b'2026-03-17T18:21:30-04:00',b'INVALID_TIMESTAMP_HERE__')]:
            with self.assertRaises(ValueError):
                prepare_upload(bad, self.events, self.ref, pd.Timedelta(0))
        pd.testing.assert_frame_equal(self.events, original)
        frame = pd.read_csv(io.BytesIO(payload));frame.loc[0,'event_id'] = None
        with self.assertRaises(ValueError):parse_events(frame)

    def test_uploaded_closure_of_all_consults(self):
        now = NOW + pd.Timedelta(minutes=1)
        rows = [{"event_id": f"CLOSE-{i}", "event_time": now, "ingested_at": now,
                 "consult_id": r.consult_id, "event_type": "note_signed", "physician_id": "",
                 "hospital_id": r.hospital_id, "description": "Synthetic note signed"}
                for i,r in enumerate(self.a['state']['consults'].itertuples())]
        after = analyze(pd.concat([self.events,parse_events(pd.DataFrame(rows))]),self.ref,now)
        self.assertTrue(after['state']['consults'].empty)
        self.assertEqual(len(after['state']['physicians']),8)
        self.assertFalse([p for p in after['problems'] if p['kind'] in {'consult','note'}])

    def scarcity_state(self):
        state = dict(self.a['state'])
        phys = state['physicians'].copy()
        phys['available_for_routing'] = False
        for pid, load in [('PHY-002',0),('PHY-003',4),('PHY-004',0)]:
            phys.loc[phys.physician_id == pid, ['available_for_routing','minutes_to_shift_end','open_consults']] = [True,90,load]
        phys.loc[phys.physician_id == 'PHY-004','minutes_to_shift_end'] = 30
        state['physicians'] = phys
        state['consults'] = pd.DataFrame([{'consult_id':'SCARCE','hospital_id':'SYN-002',
                                          'physician_id':'Unassigned','routing_failures':1}])
        state.pop('coverage',None)
        ref = dict(self.ref)
        ref['credentials'] = pd.DataFrame([('PHY-002','SYN-001'),('PHY-003','SYN-001'),
                                          ('PHY-002','SYN-002'),('PHY-004','SYN-001')],
                                          columns=['physician_id','hospital_id'])
        return state, ref

    def test_active_scarcity_ranking_and_hard_gates(self):
        state, ref = self.scarcity_state()
        c = reassign_candidates(state,ref,'SYN-001',None)
        self.assertEqual(c.physician_id.tolist(), ['PHY-003','PHY-002'])
        self.assertEqual(c.scarcity_cost.tolist(), [0,1])
        p={'kind':'consult','hospital_id':'SYN-001','stage':'Triaged','consult_id':'SCARCE-TARGET'}
        rec=recommend(p,state,ref,self.a['eligibility'])
        self.assertEqual(rec['target'],'PHY-003')
        self.assertEqual(rec['candidates'][1]['scarce_hospitals'],['SYN-002'])
        self.assertEqual(reassign_candidates(state,ref,'SYN-001','PHY-003').physician_id.tolist(),['PHY-002'])

    def test_supplied_credential_event_counterexample(self):
        new = parse_events(pd.read_csv('demo/credential_preservation_events.csv'))
        a = analyze(pd.concat([self.events,new]),self.ref,NOW)
        p = next(p for p in a['problems'] if p.get('consult_id') == 'CON-2200')
        self.assertEqual(p['rec']['target'],'PHY-003')
        first,second = p['rec']['candidates']
        self.assertEqual((first['scarcity_cost'],second['scarcity_cost']),(1,2))
        self.assertEqual(second['scarce_hospitals'],['SYN-006','SYN-008'])
        self.assertLess(second['open_consults'],first['open_consults'])

    def test_context_clocks_acknowledgment_and_history(self):
        p = dict(next(p for p in self.a['problems'] if p.get('consult_id') == 'CON-2105'))
        store=ActionStore(audit_path=None)
        store.reassign(p,'PHY-003',NOW)
        h=next(iter(store.handoffs.values()));packet=h['packet']
        self.assertAlmostEqual(packet['minutes_in_stage'],10 + 55/60)
        self.assertEqual(packet['stage_entered_at'],p['stage_entered_at'].isoformat())
        self.assertEqual(packet['packet_created_at'],NOW.isoformat())
        self.assertEqual(packet['routing_attempts'],p['routing_failures'])
        self.assertEqual(packet['routing_history'],p['routing_history'])
        failed=[e for e in packet['routing_history'] if e['event_type']=='routing_attempt_failed']
        self.assertEqual(len(failed),packet['routing_attempts'])
        self.assertTrue(all(e['event_id'] and e['event_time'] for e in failed))
        self.assertAlmostEqual(packet['remaining_deadline_min'],-55/60)
        store.tick(NOW+pd.Timedelta(minutes=2));self.assertEqual(h['status'],'overdue')
        with self.assertRaises(ValueError):store.acknowledge_handoff(h['id'],'PHY-002',NOW)
        store.acknowledge_handoff(h['id'],'PHY-003',NOW+pd.Timedelta(minutes=2))
        self.assertTrue(h['late']);self.assertEqual(h['status'],'acknowledged')
        # New Triaged->Assigned assignment starts a new legitimate stage; same/later-stage handoffs preserve clocks.
        stage_consult={'consult_id':'CLOCK','consult_type':'Stroke','current_stage':'Imaging Review',
                       'stage_entered_at':NOW-pd.Timedelta(minutes=10),'physician_id':'PHY-001',
                       'prior_physicians':['PHY-002'],'routing_failures':2}
        packet=create_task_for_stage(stage_consult,'PHY-003','Transfer',NOW)
        self.assertEqual(packet.remaining_deadline_min,2)
        self.assertEqual(packet.prior_physicians,['PHY-002','PHY-001'])
        note={**stage_consult,'current_stage':'Documentation','stage_entered_at':NOW-pd.Timedelta(minutes=5),
              'note_clock_start':NOW-pd.Timedelta(minutes=40)}
        self.assertEqual(create_task_for_stage(note,'PHY-003','Transfer',NOW).remaining_deadline_min,5)

    def test_coverage_context_and_approval_gate(self):
        a=self.a;store=ActionStore(audit_path=None)
        p=next(p for p in a['problems'] if p['rec']['action']=='split_shift')
        candidate=a['eligibility'][a['eligibility'].physician_id==p['rec']['target']].iloc[0]
        names=self.ref['hospitals'].set_index('hospital_id').hospital_name.to_dict()
        o=store.propose_split_shift(p,candidate,a['eligibility'],names,NOW,'Clinical Lead')
        self.assertFalse(store.handoffs)
        self.assertNotIn('consult_id',o['coverage_context'])
        self.assertEqual(store.respond(o['id'],True,NOW),'This offer is no longer open.')
        with self.assertRaises(ValueError):store.approve(o['id'],NOW,'Ops Lead')
        store.approve(o['id'],NOW,'Clinical Lead');store.respond(o['id'],True,NOW)
        self.assertFalse(store.handoffs)
        after=analyze(self.events,self.ref,NOW,store.accepted_shifts)
        self.assertFalse([p for p in after['problems'] if p['kind']=='gap'])

    def test_features_schedule_completed_arrivals_and_suppression(self):
        state=self.a['state'];consult=state['consults'].iloc[0].to_dict()
        physician=state['physicians'].iloc[0].to_dict();pid=physician['physician_id']
        ref=dict(self.ref);ref['on_shift']=self.ref['on_shift'].copy()
        ref['on_shift'].loc[ref['on_shift'].physician_id==pid,'shift_end_et']='19:00'
        physician['minutes_to_shift_end']=40
        feature=build_features_for_consult(consult,physician,state,ref,NOW)
        self.assertEqual([feature[f'scheduled_on_shift_{m}'] for m in (15,60,90)],[1,0,0])
        hid=consult['hospital_id'];expected=state['events'][(state['events'].hospital_id==hid)&
             (state['events'].event_type=='consult_received')&
             (state['events'].event_time>NOW-pd.Timedelta(minutes=60))].consult_id.nunique()
        self.assertEqual(feature['arrivals_last60'],expected)
        self.assertTrue(np.isnan(feature['response_history_count']))
        ml=annotate_problem(next(p for p in self.a['problems'] if p['kind']=='consult'),state,self.ref,NOW)
        self.assertEqual(ml['status'],'experimental')
        self.assertNotIn('stage_sla_probability',ml)
        self.assertEqual(experimental_pools(state,self.ref,self.a['coverage'])['status'],'unavailable')

if __name__=='__main__':unittest.main(verbosity=2)

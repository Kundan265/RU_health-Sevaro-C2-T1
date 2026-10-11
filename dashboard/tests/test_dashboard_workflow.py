"""UI regression scenarios: run directly, optionally pass a JSON evidence output path."""
from pathlib import Path
import os,json,sys,tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
_temp=tempfile.TemporaryDirectory(prefix='sevaro-ui-')
os.environ['SEVARO_AUDIT_PATH']=str(Path(_temp.name)/'audit.jsonl')
from streamlit.testing.v1 import AppTest
from engine.loader import ET
import pandas as pd
at=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=30).run()
checks=[]
def check(name,condition):
 assert condition,name
 checks.append({'check':name,'status':'pass'})
def button(label=None,key=None):
 return next(x for x in at.button if (x.label==label if label else x.key==key))
def run():
 at.run();check('no UI exception',not at.exception)
check('clean startup',not at.exception)
button('Reference 18:20 scenario').click();run()
check('reference clock',at.session_state['frozen_now']==pd.Timestamp('2026-03-17 18:20',tz=ET))
check('Clinical Lead default',at.session_state['approver_role']=='Clinical Lead')
clock=at.session_state['frozen_now'];run();check('paused rerun retains clock',clock==at.session_state['frozen_now'])
button(key='ss_gap:SYN-005').click();run()
s=at.session_state['store'];o=next(iter(s.proposals.values()))
check('pending approval only',o['status']=='pending_approval' and not s.sent_offers())
role=next(x for x in at.selectbox if x.label=='Who approves split-shift offers?')
role.select('Ops Lead');run()
check('pending offer retains its required approver',o['approver']=='Clinical Lead' and any(b.label=='Approve (Clinical Lead)' for b in at.button))
role=next(x for x in at.selectbox if x.label=='Who approves split-shift offers?')
role.select('Clinical Lead');run()
check('explicit approver configuration remains selected',at.session_state['approver_role']=='Clinical Lead')
button('Approve (Clinical Lead)').click();run()
check('approved simulated send',o['status']=='sent' and o['approved_by']=='Clinical Lead')
button('Accept').click();run()
check('accept updates coverage',o['status']=='accepted' and len(s.accepted_shifts)==1)
check('future gaps cleared',not any('Propose offer' in b.label for b in at.button))
check('staffing has no fictitious consult handoff',not s.handoffs)
check('joiner protection shown in interface',any('gap hospitals only + seeded count' in m.label for m in at.metric))
check('accepted shift restricted to gap and waiting hospitals',set(s.accepted_shifts[0]['hospitals'])=={'SYN-005','SYN-008','SYN-010'})
# Reassignment and acknowledgment after supplied activity-end event injection.
button('Reference 18:20 scenario').click();run()
s=at.session_state['store']
# Free PHY-005 by supplying legitimate activity-end and status events (UI will recompute).
from engine.loader import EVENT_COLUMNS
now=pd.Timestamp('2026-03-17 18:20',tz=ET)
# Controlled scenario: availability ends are delivered as normal events to actual roster doctors.
rows=[]
for i,et in enumerate(['incoming_call_ended','outgoing_call_ended','video_ended','camera_inactive','physician_status_changed']):
 rows.append({'event_id':f'UI-FREE-{i}','event_time':now,'ingested_at':now,'consult_id':'',
              'event_type':et,'physician_id':'PHY-005','hospital_id':'','description':'Self-reported status: Available'})
at.session_state['events']=pd.concat([at.session_state['events'],pd.DataFrame(rows)[EVENT_COLUMNS]],ignore_index=True)
run()
reassign=[b for b in at.button if b.key and b.key.startswith('ra_')]
check('real problem offers reassignment',bool(reassign))
reassign[0].click();run()
check('handoff pending',bool(s.handoffs) and next(iter(s.handoffs.values()))['status']=='pending')
h=next(iter(s.handoffs.values()))
button('+5 min').click();run()
check('handoff overdue visible',h['status']=='overdue')
button(key='ack_'+h['id']).click();run()
check('handoff acknowledgment visible',h['status']=='acknowledged' and h['late'])
# Manual clock offset does not get added again on a plain rerun.
check('explicit advance',at.session_state['offset_min']==5)
results={'checks':checks,'count':len(checks),'tabs':[t.label for t in at.tabs],
         'evidence_type':'Streamlit AppTest; supplied-event injection, not browser upload',
         'handoff':h}
if len(sys.argv)>1:
 Path(sys.argv[1]).write_text(json.dumps(results,default=str,indent=2))
print(json.dumps({'checks':len(checks),'tabs':results['tabs'],'handoff_status':h['status']}))

_temp.cleanup()

# 🧪 Test Progress

# Slot (Short-term) API
| Module / Test Item | Status |
|--------------------|---------|
| add_slot    | ✅ Debugged |
| clear_container    | ✅ Debugged |
| get_container_size   | ✅ Debugged |
| filter_and_route_slots  | ✅ Debugged |
| compress_slots      | ✅ Debugged |
| transfer_slot_to_text | ✅ Debugged |

# Long-term Memory API
| Module / Test Item | Status |
|--------------------|---------|
| instantiate_sem_record    | ✅ Debugged |
| instantiate_epi_record    | ✅ Debugged |
| instantiate_proc_record   | ✅ Debugged |
| size(@property)  | ✅ Debugged |
| get_records_by_ids     | ✅ Debugged |
| get_last_k_records | ✅ Debugged |
| is_exists    | ✅ Debugged |
| add    | ✅ Debugged |
| update   | ✅ Debugged |
| batch_memory_process  | ✅ Debugged |
| delete    | ✅ Debugged |
| query | ✅ Debugged |
| abstract_episodic_records | ❌ Not yet debugged |
| get_nearest_k_records    | ✅ Debugged |
| save   | ✅ Debugged |
| load  | ✅ Debugged |


# Test for STM and LTM
```
export OPENAI_API_KEY=""
export OPENAI_BASE_URL=""

python ltm_test.py
python stm_test.py
```

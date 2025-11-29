from functools import wraps
from memory.api.slot_process_api import SlotProcess
from src.agents.experiment_agent.sub_agents.experiment_master.workflow_state_machine import (
    WorkflowContext,
    WorkflowState,
)

def short_term_slot_trace(context: WorkflowContext):
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            result = await func(self, *args, **kwargs)
            # Debug output
            print(f"[DEBUG] Agent result type: {type(result)}")
            print(
                f"[DEBUG] Agent result: {str(result)[:200] if result else 'None'}"
            )

            # Store result in context
            self._update_context_with_result(
                context, context.current_state, result
            )
            agent_name = kwargs.get("agent_name")

            print(f"[COMPLETED] {agent_name} agent finished")

            # Save workflow snapshot after each agent execution
            self.cache_manager.save_workflow_snapshot(
                context, agent_name=agent_name, agent_output=result
            )

            if context.iteration_count >= context.max_iterations or context.current_state == WorkflowState.COMPLETED:
                working_slots = await self.slot_process.transfer_experiment_agent_context_to_working_slots(context=context, state=context.current_state)

                for slot in working_slots:
                    self.slot_process.add(slot)

                routed_slot_container = await self.slot_process.filter_and_route_slots()
                inputs = await self.slot_process.generate_long_term_memory(routed_slot_container)
                sem_record_list = []
                epi_record_list = []
                proc_record_list = []

                for i in inputs:
                    memory_type = i.get("memory_type")
                    if memory_type == "semantic":
                        sem_record = self.semantic_memory_store.instantiate_sem_record(
                            **i.get("input")
                        )
                        sem_record_list.append(sem_record)
                    elif memory_type == "episodic":
                        epi_record = self.episodic_memory_store.instantiate_epi_record(
                            **i.get("input")
                        )
                        epi_record_list.append(epi_record)
                    elif memory_type == "procedural":
                        proc_record = self.procedural_memory_store.instantiate_proc_record(
                            **i.get("input")
                        )
                        proc_record_list.append(proc_record)
                
                self.semantic_memory_store.add(sem_record_list)
                self.episodic_memory_store.add(epi_record_list)
                self.procedural_memory_store.add(proc_record_list)

                self.slot_process.clear_container()
                self.slot_process.filtered_slot_container = []
                self.slot_process.routed_slot_container = []

            return result
        return wrapper
    return decorator
                

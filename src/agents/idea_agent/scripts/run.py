import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.ligagent import LigAgent
from agent import init_logger, get_logger
import time




if __name__ == "__main__":
    # 初始化全局日志
    init_logger(filename='01.log')
    logger = get_logger()
    
    agent = LigAgent()
    agent.memory["topic"] = "Physics-Informed Neural Networks."
    agent.memory["retrieval_keywords"].append(agent.memory["topic"])
    agent.memory["background_knowledge"].append("Physics-Informed Neural Networks (PINNs) are a class of neural models that incorporate physical laws directly into the training process. The key idea is to embed governing equations—such as differential equations, conservation laws, and initial or boundary conditions—into the loss function so that the network not only fits observational data but also satisfies the underlying physics.")
    logger.info("========================================")
    logger.info("🤖 Hello, I am LigAgent!")
    logger.info(f"💡 The research topic is {agent.memory['topic']}")
    time.sleep(2)

    max_turn = 50
    for i in range(max_turn):
        logger.info("========================================")
        logger.info(f"Turn {i+1}:")
        info = {}
        # select action
        logger.info("🧠 Selecting action...")
        if len(agent.memory["steps"]) == 0:
            action = "knowledge_aquisition"
        else:
            action = agent.select_action(observation=agent.memory["steps"][-1])

        # select information from memory
        # info = agent.select_memory(action,info)

        # perform action
        agent.perform_action(action, **info)

        # import pdb; pdb.set_trace()

        # update memory
        
        
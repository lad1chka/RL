import os
import sys

# Add parent directory to path to ensure sumo_rl can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl.agents.dqn_agent import DQNAgent
from sumo_rl import SumoEnvironment

if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 1
    episodes = 1

    env = SumoEnvironment(
        net_file="./homeworks/hw1/sumo-rl/sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        route_file="./homeworks/hw1/sumo-rl/sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=False,
        num_seconds=32000,
        min_green=5,
        delta_time=5,
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()
        
        # Create agents with correct state size based on encoded states
        dqn_agents = {}
        for ts in env.ts_ids:
            # Get the actual encoded state size
            encoded_state = env.encode(initial_states[ts], ts)
            state_size = len(encoded_state)
            action_size = env.action_space.n if hasattr(env.action_space, "n") else env.action_space[ts].n
            
            dqn_agents[ts] = DQNAgent(
                state_size=state_size,
                action_size=action_size,
                gamma=gamma,
                lr=1e-3,
                buffer_size=50000,
                batch_size=64,
                target_update=500,
                learning_starts=0,
                epsilon_start=1.0,
                epsilon_end=0.01,
                epsilon_decay=0.995,
                hidden_sizes=[128, 128],
                dropout_p=0.2,
            )

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()

            # Store current states for each agent
            current_states = {ts: env.encode(initial_states[ts], ts) for ts in initial_states.keys()}
            
            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                # Get actions from each agent based on current state
                actions = {ts: dqn_agents[ts].act(current_states[ts]) for ts in dqn_agents.keys()}

                s, r, done, info = env.step(action=actions)

                # Update each agent with experience
                for agent_id in s.keys():
                    next_state = env.encode(s[agent_id], agent_id)
                    dqn_agents[agent_id].step(
                        current_states[agent_id], 
                        actions[agent_id], 
                        r[agent_id], 
                        next_state, 
                        done["__all__"]
                    )
                    current_states[agent_id] = next_state

            env.save_csv(f"outputs/4x4/ql-4x4grid_dqn_run{run}", episode)

    env.close()

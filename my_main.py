from magent2.environments import battle_v4
import os
import cv2
import torch
from torch import nn
import numpy as np
from torch_model import QNetwork
from final_torch_model import QNetwork2

def preprocess_observation(observation):
    """Chuyển đổi dữ liệu đầu vào về định dạng [channels, height, width]"""
    return torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

'''class DQNAgent(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNAgent, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(np.prod(input_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.model(x)'''

def play_and_record(env, blue_policy, red_policy, video_path, max_steps=1000, fps=35):
    frames = []
    env.reset()

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None  # Agent is done
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "blue":  # Blue agent
                obs_tensor = preprocess_observation(observation)
                with torch.no_grad():
                    action = blue_policy(obs_tensor)
            elif agent_handle == "red":  # Red agent
                obs_tensor = preprocess_observation(observation)
                with torch.no_grad():
                    action = red_policy(obs_tensor)
            else:  # Random agent
                action = env.action_space(agent).sample()

        env.step(action)

        # Record frames for the video
        if agent == "blue_0":  # Record only for one agent to save memory
            frames.append(env.render())

        if len(frames) >= max_steps:  # Stop recording after max_steps
            break

    # Save video
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    # Load environment
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 35

    # Reset environment before accessing agents
    env.reset()
    print("Agents in environment:", env.agents)

    # Load pretrained Red agent
    
    red_net = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n)
    red_net.load_state_dict(torch.load("red.pt", map_location="cpu"))
    red_net.eval()

    # Define Red policy (Pretrained)
    def red_policy(obs_tensor):
        return red_net(obs_tensor).argmax().item()

    # Load pretrained Final agent
    
    final_net = QNetwork2(env.observation_space("blue_0").shape, env.action_space("blue_0").n)
    final_net.load_state_dict(torch.load("red_final.pt", map_location="cpu"))
    final_net.eval()

    # Define Final policy
    def final_policy(obs_tensor):
        return final_net(obs_tensor).argmax().item()
    
     # Load trained DQN agent for Blue
    trained_net = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n)
    trained_net.load_state_dict(torch.load("blue.pth", map_location="cpu"))
    trained_net.eval()

    # Define Blue policy (DQN)
    def blue_policy(obs_tensor):
        return trained_net(obs_tensor).argmax().item()

    # Play and record Blue vs Random
    print("Running Blue vs Random...")
    play_and_record(env, blue_policy, lambda _: env.action_space("red_0").sample(),
                    os.path.join(vid_dir, "blue_vs_random.mp4"), max_steps=1000, fps=fps)

    # Play and record Blue vs Pretrained Red
    print("Running Blue vs Pretrained Red...")
    play_and_record(env, blue_policy, red_policy,
                    os.path.join(vid_dir, "blue_vs_red.mp4"), max_steps=1000, fps=fps)

    # Play and record Blue vs Pretrained Final
    print("Running Blue vs Pretrained Final...")
    play_and_record(env, blue_policy, final_policy,
                    os.path.join(vid_dir, "blue_vs_final.mp4"), max_steps=1000, fps=fps)

    env.close()

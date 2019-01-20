import numpy as np
from physics_sim import PhysicsSim
import math 

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, target_ang=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        
        self.runtime = runtime

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 0.]) 
        
        self.target_ang = np.array([0.,0.,0.])

    def get_reward(self):
        
        zero = 0
        
        """Uses current pose of sim to return reward."""
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        #reward = -.4*(abs(self.sim.pose[1] - self.target_pos[1])).sum()-.2*(abs(self.sim.pose[0] - self.target_pos[0])).sum()-.2*(abs(self.sim.pose[2] - self.target_pos[2])).sum() - .01 * (abs(self.sim.pose[3:] - self.target_ang).sum()) + .05 * self.sim.runtime
        
        #reward = -(abs(self.sim.pose[0] - self.target_pos[0])).sum()-(abs(self.sim.pose[1] - self.target_pos[1])).sum()- (abs(self.sim.pose[2] - self.target_pos[2])).sum()
        
        #reward = (abs(self.sim.pose[:3] - self.target_pos)).sum()
        
        #reward  = (1 - 0.01 * abs(self.sim.pose[2] - self.target_pos) - 0.01*abs(self.sim.v[2]))
        
        #reward = self.sim.runtime + (3 * (self.sim.runtime - 3))
        
        #reward = (self.sim.runtime) - (abs(self.sim.pose[:3] - self.target_pos).sum()) - (abs(self.sim.pose[2:] - self.target_ang).sum())
        
        #reward = 1.-.3*(abs(self.sim.pose[1] - self.target_pos[1])).sum() -.3*(abs(self.sim.v[1] - zero).sum())+ .001* self.sim.runtime
        
        #reward = -.4*(abs(self.sim.pose[1] - self.target_pos[1])).sum() + (.05 * self.sim.runtime)
        
        #reward = -.1*(abs(self.sim.pose[1] - self.target_pos[1])).sum() + (.1*self.sim.runtime)
        
        reward = self.sim.runtime
        
        return reward  

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0.0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            #pose_all.append(self.sim.pose)
           
            pose_all.append(self.sim.pose[2])
            
                       
        #next_state = np.concatenate(pose_all)                  
        
        next_state = np.array(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        #state = np.concatenate([self.sim.pose] * self.action_repeat)
        
        state = [self.sim.pose[2]] * self.action_repeat
        
        return state
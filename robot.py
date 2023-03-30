# Import some installed modules.
import numpy as np


# Define the Robot class
class Robot:

    # Initialisation function to create a new robot
    def __init__(self, model, max_action, goal_state):
        # STATE AND ACTION DATA
        # The maximum magnitude of the robot's action. Do not edit this.
        self.max_action = max_action

        # The goal state
        self.goal_state_saved = goal_state
        self.goal_state = goal_state
        # timestep
        self.plan_timestep = 0
        # train_planning horizon
        self.planning_horizon = 2
        # Flag to indicate whether the robot needs to plan
        self.needs_planning = True
        # steps in episode
        self.episode_num_steps=0
        # MODEL DATA
        self.model = model

        # start state
        self.start_saved = False
        self.start_state = [0,0] # initially 0,0, set during planning
        # VISUALISATION DATA
        # A list of paths that should be drawn to the screen.
        self.paths_to_draw = []
        
        self.uncertainty_threshold = 1
        # number of resets
        self.resets = 0
        self.goal_index = 0
        # test_planning horizon
        self.test_planning_horizon = 2

        self.goal_states = [np.array([0.13,0.13]), np.array([0.13,0.72]), np.array([0.72,0.72]), np.array([0.72,0.13])]
        self.goal_shifted = False
        self.circle_done = False

        self.current_path = []
        self.goal_paths = []
        self.best_distance = np.inf
        self.best_path = []

        self.test_milestone_idx = 0
    
    # Function to compute the optimal action to take, given the robot's current state.
    def planning_cross_entropy(self, current_state, planning_horizon, train=True):

        if train == True:
            iterations = 2
            k = 15
            # more samples for more diversity of paths, during training
            no_paths = 150
        
        else:
            iterations = 2
            k = 15
            no_paths = 150

        # set seed
        np.random.seed()

        mean = np.zeros(planning_horizon)+25
        cov = np.eye(planning_horizon)+100

        for i in range(iterations):
            paths = []
            scores = []
            path_actions = []
            path_angles = []

            # sample no of paths
            for j in range(no_paths):
                # Create an empty array to store the planned actions.
                planned_actions = np.zeros([planning_horizon, 2], dtype=np.float32)
                # Create an empty array to store the planned states.
                planned_states = np.zeros([planning_horizon, 2], dtype=np.float32)
                uncertainties = np.zeros([planning_horizon, 2], dtype=np.float32)
                #print("uncertainty:", np.sum(uncertainties))
                
                # Set the initial state in the planning to the robot's current state.
                planning_state = np.copy(current_state)

                # Create an empty array to store the planned angles.
                planned_angles = np.zeros([planning_horizon, 2], dtype=np.float32)

                # sample a sequence of actions from the current mean and cov
                seq = np.random.multivariate_normal(mean, cov)
                # store the angles in planned angles
                planned_angles = seq

                for m in range(planning_horizon):
                    # get angle
                    angle = seq[m]
                    #noise = np.random.normal(0,0.1)
                    # convert angle to range [0, 2*pi]
                    angle = np.fmod(angle, 2 * 3.141592) 
        
                    # convert angle to action
                    action = self.convert_angle_to_action(angle)
                    # simulate next state using the model
                    planning_state, uncertainty = self.model.predict(planning_state, action)

                    # add this action to array of planned actions
                    planned_actions[m] = action
                    # add this state to array of planned states
                    planned_states[m] = planning_state
                    uncertainties[m] = uncertainty
                
                # compute ther distances to the goal at the last state

                planned_scores = self.score(planned_states, uncertainties, train)
                # add the score and the sequence to the scores array    
                scores.append(planned_scores)

                # add the planned actions to the list of paths
                path_actions.append(planned_actions)
                # add the planned states to the list of paths
                paths.append(planned_states)
                # add the planned actions to the list of paths
                path_angles.append(planned_angles)
                # add distance to path_distances
                
            # sort the scores in ascending order (lowest to highest)
            best_path_indices = np.argsort(scores)[:k]

            # get the best k paths (lowest scores = the best scores)
            best_actions = [path_actions[i] for i in best_path_indices]
            # get the best k paths
            best_angles = [path_angles[i] for i in best_path_indices]
            # best_states = [paths[i] for i in best_path_indices]

            # find the new mean of the best k angles across the best k paths
            mean = np.mean(best_angles, axis=0)
            # find the new covariance of the best k angles across the best k paths
            cov_new = np.cov(best_angles, rowvar=False)
            # update cov with diagonal elements only
            cov = np.diag(np.diag(cov_new))

        # best sequence angles is the first sequence in the sorted sequences
        best_sequence_angles = best_angles[0]
        
        return best_actions[0]
    
    
    def next_action_mixed_loop_test(self,state):
        self.goal_state = self.goal_states[self.test_milestone_idx]
        # print("the next milestone is", self.goal_states[self.test_milestone_idx])
        if (np.linalg.norm(state-self.goal_state) < 0.07) and (np.array_equal(self.goal_state, self.goal_state_saved) == False):
            self.test_milestone_idx += 1
            self.needs_planning = True

        if self.needs_planning:
            #print("the milestones found for testing are:", self.goal_states)
            self.planned_actions = self.planning_cross_entropy(state, self.test_planning_horizon, train=False)
            # Set the flag so that the robot knows it has already done the planning.
            self.needs_planning = False
            # Reset the timestep in the plan.
            self.plan_timestep = 0
        
        # Check if the robot has any more actions left to execute in the plan.
        if self.plan_timestep < len(self.planned_actions):
                # If there are more actions, return the next action.
                next_action = self.planned_actions[self.plan_timestep]
                # Increment the timestep in the plan.
                self.plan_timestep += 1

                # If there are no actions left in the plan, then replan on the next step
                if self.plan_timestep == len(self.planned_actions):
                        self.needs_planning = True
        
        self.episode_num_steps += 1
        # print("Episode num steps: ", self.episode_num_steps)
        # Return the next action
        return next_action


    def score(self, planned_states, uncertainties, train=True):
        score = 0
        distance_score = np.linalg.norm(planned_states[-1] - self.goal_state)
        uncertainty_score = np.sum(uncertainties)
        if train == True:

            if self.circle_done == True:
                # encourage more variation once the general path has been determined, to try refine the best path.
                score = distance_score - 2 * uncertainty_score # bigger uncertainty = better score
            else:
                score = distance_score
        else:
            # focus less on certainty at the very start and very end.
            if np.linalg.norm(planned_states[-1] - self.goal_state_saved) <= 0.2 or self.test_milestone_idx == 0:
                score = distance_score
            else:
                # factor in certainty of path during the middle of the robot's path as test time
                score = distance_score + 6 * uncertainty_score

        return score 
    
    def update_milestones(self, first=True):
        if first == True:
            # On first, explore map around the corners
            # 1. Replace the nearest corner to the goal state, with the goal state
            goal_distances = [np.linalg.norm(i - self.goal_state_saved) for i in self.goal_states]
            closest_corner_to_goal = np.argmin(goal_distances)
            self.goal_states[closest_corner_to_goal] = self.goal_state_saved
            # 2. Set the next goal state (milestone) Find corner closest to goal state
            start_distances = [np.linalg.norm(i - self.start_state) for i in self.goal_states]
            closest_to_start = np.argmin(start_distances)
            self.goal_states[closest_to_start] = self.start_state
            self.goal_index = closest_to_start + 1
            self.goal_state = self.goal_states[(self.goal_index) % 4]
        
        else:
            # Circle completed. Find milestones along the best path.
            # 1. Go through goal_paths, and work out distance of each path
            updated = False
            for (idx, path) in enumerate(self.goal_paths):
                path_length = len(path)
                path_distance = 0
                for x in range(len(path)-1):
                    path_distance += np.linalg.norm(path[x+1]-path[x])
                
                # factor of 200 for the "closeness" empirically found to select the better path
                score = path_distance - 200*(path_distance / path_length)
                # print("Score of path is:", score)
                
                if score < self.best_distance:
                    # update the best path
                    self.best_distance = score
                    self.best_path = self.goal_paths[idx]
                    updated = True
            
            self.goal_paths = []
            self.goal_paths.append(self.best_path)

            # 2. Update the goal states to the milestones
            self.goal_states = []
            if updated == True:
                for i, point in enumerate(self.best_path):
                    if i % 10 == 0 and i != 0:
                        self.goal_states.append(point)
            self.goal_index = 0
            self.goal_states.append(self.goal_state_saved)
            self.goal_states.insert(0, self.best_path[9]) 
            # print("The new milestones are", self.goal_states)

    def next_action_mixed_loop_train(self,state):

        if self.start_saved == False:
            # save the start state coordinates at the very beginning
            self.start_state = state
            self.start_saved = True
            # first = True, i.e. robot firsts explores in a circle, looking to find better semi-circle path from state->goal
            self.update_milestones(first=True) 
            # print("the goal states list is", self.goal_states)
            # print("the current goal state is", self.goal_state)
        
        if np.linalg.norm(state-self.goal_state) < 0.03:
            self.goal_index += 1
            reset = False
            next_action = [0,0]
            self.plan_timestep = 0
            self.needs_planning = True
            self.episode_num_steps = 0


            if self.circle_done == False:
                # 1. Work out if current path needs to be wiped
                if np.array_equal(self.goal_state, self.goal_state_saved):
                    # reset the current path
                    self.goal_paths.append(self.current_path)
                    self.current_path = []
                    # print("Found goal state")
                    # print("Current path added!\n")
                    
                if np.array_equal(self.goal_state, self.start_state):
                    #print("before: ", self.current_path)
                    self.current_path.reverse()
                    #print("after reversing: ", self.current_path)
                    #print("")
                    self.goal_paths.append(self.current_path)
                    self.current_path = []
                    self.circle_done = True
                    self.update_milestones(first = False)
                    #print("Found start state, circle is now done\n")
                
                #else:
                    #print("Found milestone.")
                    #print("")

                self.goal_state = self.goal_states[self.goal_index % 4]

            else:
                reset = False
                if np.array_equal(self.goal_state, self.goal_state_saved):
                    reset = True
                    self.goal_paths.append(self.current_path)
                    self.update_milestones()

                # go to next milestone
                self.goal_state = self.goal_states[self.goal_index % len(self.goal_states)] 

        else:
            reset=False
            self.current_path.append(state)
            if self.needs_planning:
                # Do some planning.
                self.planned_actions = self.planning_cross_entropy(state, self.planning_horizon)
                # Set the flag so that the robot knows it has already done the planning.
                self.needs_planning = False
                # Reset the timestep in the plan.
                self.plan_timestep = 0
            
            # Check if the robot has any more actions left to execute in the plan.
            if self.plan_timestep < len(self.planned_actions):
                # If there are more actions, return the next action.
                next_action = self.planned_actions[self.plan_timestep]
                # Increment the timestep in the plan.
                self.plan_timestep += 1

                # If there are no actions left in the plan, then replan on the next step
                if self.plan_timestep == len(self.planned_actions):
                    self.needs_planning = True
                    #self.train_planning_horizon+=5

        self.episode_num_steps += 1
        #print("Episode num steps: ", self.episode_num_steps)
        # Return the next action
        return next_action,reset
    
        # Function to convert a scalar angle, to an action parameterised by a 2-dimensional [x, y] direction.
    def convert_angle_to_action(self, angle):
        action_x = self.max_action * np.cos(angle)
        action_y = self.max_action * np.sin(angle)
        action = np.array([action_x, action_y])
        return action
    
    # Function to compute the next action, during the training phase.
    def next_action_training(self, state):
        # For now, just a random action. Try to do better than this!
        next_action, reset = self.next_action_mixed_loop_train(state)
        #reset = False
        return next_action, reset

    # Function to compute the next action, during the testing phase.
    def next_action_testing(self, state):
        # For now, just a random action. Try to do better than this!
        next_action = self.next_action_mixed_loop_test(state)
        return next_action

    # Function to compute the next action.
    def random_action(self):
        # Choose a random action.
        action_x = np.random.uniform(-self.max_action, self.max_action)
        action_y = np.random.uniform(-self.max_action, self.max_action)
        action = np.array([action_x, action_y])
        # Return this random action.
        return action
    

    def process_transition(self, state, action):
        self.model.update_uncertainty(state, action)

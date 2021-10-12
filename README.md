**Reinforcement Learning on a GridWorld**

**Environment.** An NxN integer gridworld is presented representing a resource mining environment (Class quarry) in which a robot agent begins at a randomised location which represents its base. There are N//2 collectable resources (with reward value of 10)which disappear from the environment when they have been collected. In addition there
are N//2 insurmountable obstacles (with reward value of -5) distributed throughout (which also form the borders of the environment) and the agent is tasked with collecting
resources and depositing them back to base (for a bumper reward equal to the square ofthe value of the resources delivered) until a specified target value of resource (10*N//4)
has been safely delivered (the termination condition). Time steps are limited to 100. Anumpy matrix is used to represent the gridworld, with a randomised index for each coordinate to demonstrate that the representation of states is immaterial. The reset function re-creates a new randomised environment and resets all the attributes as relevant.

**Markov Decision Process (MDP)**. The agent will follow a MDP to reach its goal. An MDP is a discrete-time stochastic control process. At each time step, the agent finds itself in a state s and is able to choosean action a (up, down, left, right). At the next step, the agent moves to a new state s’ (which is the same as s in case of hitting an obstacle) and receives a reward Ra(s, s’). The probability of changing state from s to s’ is guided by the action selected, specifically the state transition function Pa(s, s’). Accordingly, s’ is dependent on s and a however it is independent of all previous s and a. This means that it satisfies the Markov property.

**Stochasticity** is introduced through a randomised 0.3 chance that resources are contaminated, which is only discovered when collected by the agent, and in this case the
entire load of resources carried by the agent at that time are also contaminated and have to be dumped (resources remain in place in this case). However resources which
have already been delivered back to base are remain safely registered. There is therefore a fundamental trade-off for the agent to calculate of how frequently to return to
base, given that rewards are diminished by 1 for each time step – it’s potentially more efficient to spend longer collecting resources and then delivering them in bigger batches
however this also increases the risk of contamination.

**Agent.** The agent itself has the attributes of resource_load (measure of value of current resources carried); resource_delivered (measure of value of current resources delivered back to base) and resource_target (which is set as 10N//4 to ensure that it’s possible to terminate even in the case of contamination, as the intention was to keep the
environment workable). As well as these attributes, the agent has available to it the knowledge of its location in the grid world and a proximity matrix of size 5x5 so it can
learn the types of location that surround it. For the q-learning implementation , although the resource target is used to assess the termination condition, the state
consists only of the information about the current location of the agent and the total number of different states that the agent can reach is finite (the other attributes and
proximity information are only used for the proximal policy optimisation deep learning implementation ). The actions available to the agent are up, down, left and right. Resources are collected automatically when entering a resource location and automatically deposited on returning to base. Note that the agent receives an additional
reward equal to square of the resources collected on the occasion of actually delivering the resources to base (in addition to the rewards it receives for picking up rewards in first place.)

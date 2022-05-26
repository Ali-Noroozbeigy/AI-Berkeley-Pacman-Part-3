# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()

        currentStepValues= util.Counter() # the updated values will be stored in this dict


        # V0 of every state is zero
        for state in states:
            self.values[state] = 0

        for i in range(self.iterations):
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                maxSum = -10000
                for action in actions:
                    probs = self.mdp.getTransitionStatesAndProbs(state, action)
                    sumOfProbValues = 0.0
                    for prob in probs:
                        nextState, prb = prob
                        sumOfProbValues += (prb * (self.mdp.getReward(state, action, nextState) + (self.discount * self.values[nextState])))
                    maxSum = max(maxSum, sumOfProbValues)
                    if maxSum != -10000:
                        currentStepValues[state] = maxSum
            for state in states:
                # prepare for the next iteration
                self.values[state] = currentStepValues[state]



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        probs = self.mdp.getTransitionStatesAndProbs(state, action)
        sumOfProbValues = 0.0
        for prob in probs:
            nextState, prb = prob
            sumOfProbValues += (
                        prb * (self.mdp.getReward(state, action, nextState) + (self.discount * self.values[nextState])))

        return sumOfProbValues
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        bestAction = None
        maxValue = -10000

        if self.mdp.isTerminal(state):
            return bestAction

        actions = self.mdp.getPossibleActions(state)

        for action in actions:
            QValue = self.computeQValueFromValues(state,action)

            if QValue > maxValue:
                maxValue = QValue
                bestAction = action

        return bestAction

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        states = self.mdp.getStates()

        currentStepValues = util.Counter()  # the updated values will be stored in this

        # V0 of every state is zero
        for state in states:
            self.values[state] = 0


        n = len(states)
        j = 0
        for i in range(self.iterations):
            state = states[j % n]
            j += 1
            actions = self.mdp.getPossibleActions(state)
            maxSum = -10000
            for action in actions:
                probs = self.mdp.getTransitionStatesAndProbs(state, action)
                sumOfProbValues = 0.0
                for prob in probs:
                    nextState, prb = prob
                    sumOfProbValues += (prb * (self.mdp.getReward(state, action, nextState) + (
                                self.discount * self.values[nextState])))
                maxSum = max(maxSum, sumOfProbValues)
                if maxSum != -10000:
                    currentStepValues[state] = maxSum
            for state in states:
                # prepare for the next iteration
                self.values[state] = currentStepValues[state]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        statePredec = {}

        priorityQueue = util.PriorityQueue()

        for state in states:
            if not self.mdp.isTerminal(state):
                for action in self.mdp.getPossibleActions(state):
                    for tsp in self.mdp.getTransitionStatesAndProbs(state, action):
                        if tsp[0] in statePredec:
                            statePredec[tsp[0]].add(state)
                        else:
                            statePredec[tsp[0]] = set()
                            statePredec[tsp[0]].add(state)

        for state in states:
            #maxQ = -10000
            #q = 0
            qList = []
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    q = self.getQValue(state, action)
                    qList.append(q)
                    #if q > maxQ:
                        #q = maxQ
                diff = abs(self.values[state] - max(qList))
                priorityQueue.push(state, diff * -1)

        for i in range(self.iterations):
            if priorityQueue.isEmpty():
                break
            s = priorityQueue.pop()
            if not self.mdp.isTerminal(s):
                qList = []
                actions = self.mdp.getPossibleActions(s)
                for action in actions:
                    q = self.getQValue(s, action)
                    qList.append(q)
                self.values[s] = max(qList)

            for predec in statePredec[s]:
                if not self.mdp.isTerminal(s):
                    qList = []
                    actions = self.mdp.getPossibleActions(predec)
                    for action in actions:
                        q = self.getQValue(predec, action)
                        qList.append(q)
                    diff = abs(self.values[predec] - max(qList))

                    if diff > self.theta:
                        priorityQueue.update(predec, diff * -1)

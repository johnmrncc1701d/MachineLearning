package myProj;

import burlap.behavior.policy.*;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.QLearning;
import burlap.behavior.singleagent.planning.Planner;
import burlap.behavior.singleagent.planning.stochastic.DynamicProgramming;
import burlap.behavior.singleagent.planning.stochastic.policyiteration.PolicyIteration;
import burlap.behavior.singleagent.planning.stochastic.valueiteration.ValueIteration;
import burlap.domain.singleagent.graphdefined.GraphDefinedDomain;
import burlap.domain.singleagent.graphdefined.GraphStateNode;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.common.ConstantStateGenerator;
import burlap.mdp.auxiliary.common.SinglePFTF;
import burlap.mdp.auxiliary.stateconditiontest.TFGoalCondition;
import burlap.mdp.core.TerminalFunction;
import burlap.mdp.core.action.Action;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.state.NullState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.SADomain;
import burlap.mdp.singleagent.common.GoalBasedRF;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.model.RewardFunction;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.statehashing.HashableState;
import burlap.statehashing.simple.SimpleHashableStateFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

public class PlotTest {

    public static void GridWorldExample()
    {
        GridWorldDomain gw = new GridWorldDomain(11,11); //11x11 grid world
        gw.setMapToFourRooms(); //four rooms layout
        gw.setProbSucceedTransitionDynamics(0.8); //stochastic transitions with 0.8 success rate

        //ends when the agent reaches a location
        final TerminalFunction tf = new SinglePFTF(
                PropositionalFunction.findPF(gw.generatePfs(), GridWorldDomain.PF_AT_LOCATION));

        //reward function definition
        final RewardFunction rf = new GoalBasedRF(new TFGoalCondition(tf), 5., -0.1);

        gw.setTf(tf);
        gw.setRf(rf);


        final OOSADomain domain = gw.generateDomain(); //generate the grid world domain

        //setup initial state
        GridWorldState s = new GridWorldState(new GridAgent(0, 0),
                new GridLocation(10, 10, "loc0"));



        //initial state generator
        final ConstantStateGenerator sg = new ConstantStateGenerator(s);


        //set up the state hashing system for looking up states
        final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();


        /**
         * Create factory for Q-learning agent
         */
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "Q-learning";
            }

            public LearningAgent generateAgent() {
                return new QLearning(domain, 0.99, hashingFactory, 0.3, 0.1);
            }
        };

        //define learning environment
        SimulatedEnvironment env = new SimulatedEnvironment(domain, sg);

        //define experiment
        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
                10, 100, qLearningFactory);

        exp.setUpPlottingConfiguration(500, 250, 2, 1000,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
                PerformanceMetric.AVERAGE_EPISODE_REWARD);


        //start experiment
        exp.startExperiment();
    }

    public static void GridIter(SADomain domain, SimpleHashableStateFactory hashingFactory, ConstantStateGenerator sg, final String version)
    {
        Planner planner;
        if (version.equals("Value Iteration")) {
            planner = new ValueIteration(domain, 0.59, hashingFactory, 0.001, 100);
        }
        else {
            planner = new PolicyIteration(domain, 0.59, hashingFactory, 0.001, 100, 100);
        }
        DynamicProgramming lastDP = (DynamicProgramming)planner;
        long startTime = System.currentTimeMillis();
        Policy p = planner.planFromState(sg.generateState());
        long timeElapsed = System.currentTimeMillis() - startTime;
        System.out.println(version + "; Time: " + ((float)timeElapsed / 1000.0f));

        try {
            FileWriter file = new FileWriter("D:\\School\\GT\\MachineLearning\\MarkovDecisionProcess\\GridStates_" + version.replace(' ','_') + ".csv");
            for (State stt : lastDP.getAllStates()) { //JMR
                file.write("" + stt.toString().replace('\n',' ') + "," + p.action(stt) + "\n");
                //System.out.println(version + "; State: " + stt.toString() + "; Action: " + p.action(stt));
            }
            file.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    public static void MyGridWorld()
    {
        int max_size = 501;
        int cur_size = 288;
        float scale = (float)cur_size/(float)max_size;

        GridWorldDomain gw = new GridWorldDomain(cur_size,cur_size); //501, 407, 288

        gw.horizontalWall(0,(int)(60.0f * (scale)),(int)(167.0f * (scale)));
        gw.horizontalWall((int)(60.0f * (scale)) + 3,(int)(270.0f * (scale)),(int)(167.0f * (scale)));
        gw.horizontalWall((int)(270.0f * (scale)) + 3,(int)(405.0f * (scale)),(int)(167.0f * (scale)));
        gw.horizontalWall((int)(405.0f * (scale)) + 3,cur_size - 1,(int)(167.0f * (scale)));

        gw.horizontalWall(0,(int)(40.0f * (scale)),(int)(333.0f * (scale)));
        gw.horizontalWall((int)(40.0f * (scale)) + 3,(int)(280.0f * (scale)),(int)(333.0f * (scale)));
        gw.horizontalWall((int)(280.0f * (scale)) + 3,(int)(415.0f * (scale)),(int)(333.0f * (scale)));
        gw.horizontalWall((int)(415.0f * (scale)) + 3,cur_size - 1,(int)(333.0f * (scale)));

        gw.verticalWall(0,(int)(70.0f * (scale)),(int)(167.0f * (scale)));
        gw.verticalWall((int)(70.0f * (scale)) + 3,(int)(260.0f * (scale)),(int)(167.0f * (scale)));
        gw.verticalWall((int)(260.0f * (scale)) + 3,(int)(400.0f * (scale)),(int)(167.0f * (scale)));
        gw.verticalWall((int)(400.0f * (scale)) + 3,cur_size - 1,(int)(167.0f * (scale)));

        gw.verticalWall(0,(int)(50.0f * (scale)),(int)(333.0f * (scale)));
        gw.verticalWall((int)(50.0f * (scale)) + 3,(int)(290.0f * (scale)),(int)(333.0f * (scale)));
        gw.verticalWall((int)(290.0f * (scale)) + 3,(int)(425.0f * (scale)),(int)(333.0f * (scale)));
        gw.verticalWall((int)(425.0f * (scale)) + 3,cur_size - 1,(int)(333.0f * (scale)));

        gw.setProbSucceedTransitionDynamics(0.9); //stochastic transitions with 0.9 success rate

        //ends when the agent reaches a location
        final TerminalFunction tf = new SinglePFTF(
                PropositionalFunction.findPF(gw.generatePfs(), GridWorldDomain.PF_AT_LOCATION));

        //reward function definition
        final RewardFunction rf = new GoalBasedRF(new TFGoalCondition(tf), 500., -0.0005);

        gw.setTf(tf);
        gw.setRf(rf);


        final OOSADomain domain = gw.generateDomain(); //generate the grid world domain

        //setup initial state
        GridWorldState s = new GridWorldState(new GridAgent(0, 0), new GridLocation(cur_size - 1, cur_size - 1, "loc0"));


        //initial state generator
        final ConstantStateGenerator sg = new ConstantStateGenerator(s);


        //set up the state hashing system for looking up states
        final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

        GridIter(domain, hashingFactory, sg, "Value Iteration");
        //GridIter(domain, hashingFactory, sg, "Policy Iteration");

        /**
         * Create factory for Q-learning agent
         */
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "Q-learning";
            }

            public LearningAgent generateAgent() {
                QLearning q = new QLearning(domain, 0.59, hashingFactory, 0.3, 0.1);
                //q.setLearningPolicy(new EpsilonGreedy(q, 0.1));
                //q.setLearningPolicy(new GreedyQPolicy(q));
                //q.setLearningPolicy(new GreedyDeterministicQPolicy(q));
                q.setLearningPolicy(new BoltzmannQPolicy(q, 0.01));
                return q;
            }
        };

        //define learning environment
        SimulatedEnvironment env = new SimulatedEnvironment(domain, sg);

        //define experiment
        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
                1, 90, qLearningFactory);

        exp.setUpPlottingConfiguration(500, 250, 2, 1000,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
                PerformanceMetric.AVERAGE_EPISODE_REWARD);


        //start experiment
        //exp.startExperiment();
    }

    private static final int actionRebreed = 0;
    private static final int actionDontFish = 1;
    private static final int actionFishLow = 2;
    private static final int actionFishMed = 3;
    private static final int actionFishBig = 4;

    private static class FishRewardFunction implements RewardFunction {
        static int TotalReward = 0;

        @Override
        public double reward(State s, Action a, State sprime) {
            if (a.actionName().equals("action" + actionRebreed))
            {
                TotalReward += -200;
                //System.out.println("Re-breed: " + TotalReward);
                return -200;
            }
            if (a.actionName().equals("action" + actionFishLow))
            {
                TotalReward += 5;
                //System.out.println("Fish: Low Population: " + TotalReward);
                return 5;
            }
            if (a.actionName().equals("action" + actionFishMed))
            {
                TotalReward += 10;
                //System.out.println("Fish: Medium Population: " + TotalReward);
                return 10;
            }
            if (a.actionName().equals("action" + actionFishBig))
            {
                TotalReward += 50;
                //System.out.println("Fish: High Population: " + TotalReward);
                return 50;
            }
            //System.out.println("Don't Fish: " + TotalReward);
            return 0;
        }

    }

    public static void FishIter(SADomain domain, SimpleHashableStateFactory hashingFactory, ConstantStateGenerator sg, final String version)
    {
        Planner planner;
        if (version.equals("Value Iteration")) {
            planner = new ValueIteration(domain, 0.59, hashingFactory, 0.001, 100);
        }
        else {
            planner = new PolicyIteration(domain, 0.59, hashingFactory, 0.001, 100, 100);
        }
        DynamicProgramming lastDP = (DynamicProgramming)planner;
        long startTime = System.currentTimeMillis();
        Policy p = planner.planFromState(sg.generateState());
        long timeElapsed = System.currentTimeMillis() - startTime;
        System.out.println(version + "; Time: " + ((float)timeElapsed / 1000.0f));

        for (State stt : lastDP.getAllStates()) {
            System.out.println(version + "; State: " + stt.toString() + "; Action: " + p.action(stt));
        }
    }

    public static void FishWorld()
    {
        GraphDefinedDomain gw = new GraphDefinedDomain();
        gw.setTransition(0, actionRebreed, 1, 1.0);
        gw.setTransition(1, actionFishLow, 0, 0.75);
        gw.setTransition(1, actionFishLow, 1, 0.25);
        gw.setTransition(1, actionDontFish, 2, 0.7);
        gw.setTransition(1, actionDontFish, 1, 0.3);
        gw.setTransition(2, actionFishMed, 1, 0.75);
        gw.setTransition(2, actionFishMed, 2, 0.25);
        gw.setTransition(2, actionDontFish, 3, 0.75);
        gw.setTransition(2, actionDontFish, 2, 0.25);
        gw.setTransition(3, actionFishBig, 2, 0.6);
        gw.setTransition(3, actionFishBig, 3, 0.4);
        gw.setTransition(3, actionDontFish, 2, 0.05);
        gw.setTransition(3, actionDontFish, 3, 0.95);

        //ends when the agent reaches a location
        final TerminalFunction tf = new TerminalFunction() {
            @Override
            public boolean isTerminal(State s) {
                return FishRewardFunction.TotalReward >= 10000 || FishRewardFunction.TotalReward <= -10000;
            }
        };

        final RewardFunction rf = new FishRewardFunction();

        gw.setRf(rf);
        gw.setTf(tf);

        final SADomain domain = gw.generateDomain();
        State s = new GraphStateNode(0);
        final ConstantStateGenerator sg = new ConstantStateGenerator(s)
        {
            @Override
            public State generateState() {
                FishRewardFunction.TotalReward = 0;
                return src.copy();
            }
        };

        //set up the state hashing system for looking up states
        final SimpleHashableStateFactory hashingFactory = new SimpleHashableStateFactory();

        //FishIter(domain, hashingFactory, sg, "Value Iteration");
        //FishIter(domain, hashingFactory, sg, "Policy Iteration");

        /**
         * Create factory for Q-learning agent
         */
        LearningAgentFactory qLearningFactory = new LearningAgentFactory() {

            public String getAgentName() {
                return "Q-Learning";
            }

            public LearningAgent generateAgent() {
                QLearning q = new QLearning(domain, 0.59, hashingFactory, 0.3, 0.1);
                //q.setLearningPolicy(new EpsilonGreedy(q, 0.1));
                //q.setLearningPolicy(new GreedyQPolicy(q));
                //q.setLearningPolicy(new GreedyDeterministicQPolicy(q));
                q.setLearningPolicy(new BoltzmannQPolicy(q, 0.1));
                return q;
            }
        };

        //define learning environment
        SimulatedEnvironment env = new SimulatedEnvironment(domain, sg);

        //define experiment
        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(env,
                1, 100, qLearningFactory);

        exp.setUpPlottingConfiguration(500, 250, 2, 1000,
                TrialMode.MOST_RECENT_AND_AVERAGE,
                PerformanceMetric.CUMULATIVE_STEPS_PER_EPISODE,
                PerformanceMetric.AVERAGE_EPISODE_REWARD);

        //start experiment
        exp.startExperiment();
    }

    public static void main(String [] args){

        //GridWorldExample();
        //FishWorld();
        MyGridWorld();
    }
}

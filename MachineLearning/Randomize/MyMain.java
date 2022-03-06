package opt.test;

import java.io.File;
import java.util.Arrays;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import util.linalg.Vector;

import dist.DiscreteUniformDistribution;
import dist.DiscreteDependencyTree;
import dist.FixedDistribution;
import dist.AbstractDistribution;
import opt.EvaluationFunction;
import opt.OptimizationAlgorithm;
import opt.DiscreteChangeOneNeighbor;
import opt.GenericHillClimbingProblem;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.FourPeaksEvaluationFunction;
import opt.example.FlipFlopEvaluationFunction;
import opt.example.KnapsackEvaluationFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.FitnessOrderedGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import shared.ConvergenceTrainer;

import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.Adam;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.activation.LogisticSigmoid;
import shared.SumOfSquaresError;
import shared.DataSet;
import shared.Instance;
import shared.FixedIterationTrainer;
import shared.reader.DataSetReader;
import shared.reader.CSVDataSetReader;
import shared.filt.LabelSplitFilter;
import shared.filt.DiscreteToBinaryFilter;
import opt.example.NeuralNetworkOptimizationProblem;

public class MyMain {

    private static final Random random = new Random();
    private static final int NUM_ITEMS = 40;
    private static final int COPIES_EACH = 4;
    private static final double MAX_VALUE = 50;
    private static final double MAX_WEIGHT = 50;
    private static final double MAX_KNAPSACK_WEIGHT =
            MAX_WEIGHT * NUM_ITEMS * COPIES_EACH * .4;

    public static void Train(String outputFile, EvaluationFunction ef, OptimizationAlgorithm oa, int iterations)
    {
        ef.resetFunctionEvaluationCount();
        ConvergenceTrainer fit = new ConvergenceTrainer(oa);
        try {
            FileWriter file = new FileWriter(outputFile);

            file.write("iters,fevals,fitness\n");
            for (int i = 0; i < iterations; ++i)
            {
                fit.train();
                file.write("" + i + "," + (ef.getFunctionEvaluations()-i) + "," + ef.value(oa.getOptimal()) + "\n");
            }

            file.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    public static void Train_HillClimb_FourPeaks(AbstractDistribution odd, int[] n, int threshold)
    {
        DiscreteChangeOneNeighbor nf = new DiscreteChangeOneNeighbor(n);

        FourPeaksEvaluationFunction ef_c1 = new FourPeaksEvaluationFunction(threshold);
        GenericHillClimbingProblem hcp_c1 = new GenericHillClimbingProblem(ef_c1, odd, nf);

        RandomizedHillClimbing rhc_c1 = new RandomizedHillClimbing(hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\RandHillClimb\\RandomizedHillClimbing-FourPeaks_" + threshold + ".csv",
                ef_c1, rhc_c1, 100000);
    }

    public static void Train_HillClimb_FlipFlop(AbstractDistribution odd, int[] n)
    {
        DiscreteChangeOneNeighbor nf = new DiscreteChangeOneNeighbor(n);

        FlipFlopEvaluationFunction ef_c1 = new FlipFlopEvaluationFunction();
        GenericHillClimbingProblem hcp_c1 = new GenericHillClimbingProblem(ef_c1, odd, nf);

        RandomizedHillClimbing rhc_c1 = new RandomizedHillClimbing(hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\RandHillClimb\\RandomizedHillClimbing-FlipFlop_.csv",
                ef_c1, rhc_c1, 100000);
    }

    public static void Train_HillClimb_Knapsack(AbstractDistribution odd, int[] n, double[] values, double[] weights, double maximumValue, int[] copiesPerElement)
    {
        DiscreteChangeOneNeighbor nf = new DiscreteChangeOneNeighbor(n);
        KnapsackEvaluationFunction ef_c1 = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copiesPerElement);
        GenericHillClimbingProblem hcp_c1 = new GenericHillClimbingProblem(ef_c1, odd, nf);

        RandomizedHillClimbing rhc_c1 = new RandomizedHillClimbing(hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\RandHillClimb\\RandomizedHillClimbing-Knapsack_.csv",
                ef_c1, rhc_c1, 100000);
    }

    public static void Train_SimAnneal_FourPeaks(AbstractDistribution odd, int[] n, int threshold)
    {
        DiscreteChangeOneNeighbor nf = new DiscreteChangeOneNeighbor(n);

        FourPeaksEvaluationFunction ef_c1 = new FourPeaksEvaluationFunction(threshold);
        GenericHillClimbingProblem hcp_c1 = new GenericHillClimbingProblem(ef_c1, odd, nf);

        SimulatedAnnealing rhc_c1 = new SimulatedAnnealing(1E11, .95, hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\SimAnneal\\SimulatedAnnealing-FourPeaks_" + threshold + ".csv",
                ef_c1, rhc_c1, 100000);
    }

    public static void Train_SimAnneal_FlipFlop(AbstractDistribution odd, int[] n)
    {
        DiscreteChangeOneNeighbor nf = new DiscreteChangeOneNeighbor(n);

        FlipFlopEvaluationFunction ef_c1 = new FlipFlopEvaluationFunction();
        GenericHillClimbingProblem hcp_c1 = new GenericHillClimbingProblem(ef_c1, odd, nf);

        SimulatedAnnealing rhc_c1 = new SimulatedAnnealing(1E11, .95, hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\SimAnneal\\SimulatedAnnealing-FlipFlop_.csv",
                ef_c1, rhc_c1, 100000);
    }

    public static void Train_SimAnneal_Knapsack(AbstractDistribution odd, int[] n, double[] values, double[] weights, double maximumValue, int[] copiesPerElement)
    {
        DiscreteChangeOneNeighbor nf = new DiscreteChangeOneNeighbor(n);
        KnapsackEvaluationFunction ef_c1 = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copiesPerElement);
        GenericHillClimbingProblem hcp_c1 = new GenericHillClimbingProblem(ef_c1, odd, nf);

        SimulatedAnnealing rhc_c1 = new SimulatedAnnealing(1E11, .95, hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\SimAnneal\\SimulatedAnnealing-Knapsack_.csv",
                ef_c1, rhc_c1, 100000);
    }

    public static void Train_Genetic_FourPeaks(AbstractDistribution odd, int[] n, int threshold)
    {
        DiscreteChangeOneMutation mf = new DiscreteChangeOneMutation(n);
        SingleCrossOver cf = new SingleCrossOver();

        FourPeaksEvaluationFunction ef_c1 = new FourPeaksEvaluationFunction(threshold);
        GenericGeneticAlgorithmProblem hcp_c1 = new GenericGeneticAlgorithmProblem(ef_c1, odd, mf, cf);

        StandardGeneticAlgorithm rhc_c1 = new StandardGeneticAlgorithm(200, 100, 10, hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\Genetic\\GeneticAlgo-FourPeaks_" + threshold + ".csv",
                ef_c1, rhc_c1, 100000);
    }

    public static void Train_Genetic_FlipFlop(AbstractDistribution odd, int[] n)
    {
        DiscreteChangeOneMutation mf = new DiscreteChangeOneMutation(n);
        SingleCrossOver cf = new SingleCrossOver();

        FlipFlopEvaluationFunction ef_c1 = new FlipFlopEvaluationFunction();
        GenericGeneticAlgorithmProblem hcp_c1 = new GenericGeneticAlgorithmProblem(ef_c1, odd, mf, cf);

        StandardGeneticAlgorithm rhc_c1 = new StandardGeneticAlgorithm(200, 100, 10, hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\Genetic\\GeneticAlgo-FlipFlop_.csv",
                ef_c1, rhc_c1, 100000);
    }

    public static void Train_Genetic_Knapsack(AbstractDistribution odd, int[] n, double[] values, double[] weights, double maximumValue, int[] copiesPerElement)
    {
        DiscreteChangeOneMutation mf = new DiscreteChangeOneMutation(n);
        SingleCrossOver cf = new SingleCrossOver();
        KnapsackEvaluationFunction ef_c1 = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copiesPerElement);
        GenericGeneticAlgorithmProblem hcp_c1 = new GenericGeneticAlgorithmProblem(ef_c1, odd, mf, cf);

        StandardGeneticAlgorithm rhc_c1 = new StandardGeneticAlgorithm(200, 100, 10, hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\Genetic\\GeneticAlgo-Knapsack_.csv",
                ef_c1, rhc_c1, 100000);
    }

    public static void Train_MIMIC_FourPeaks(AbstractDistribution odd, int[] n, int threshold)
    {
        DiscreteDependencyTree df = new DiscreteDependencyTree(.1, n);

        FourPeaksEvaluationFunction ef_c1 = new FourPeaksEvaluationFunction(threshold);
        GenericProbabilisticOptimizationProblem hcp_c1 = new GenericProbabilisticOptimizationProblem(ef_c1, odd, df);

        MIMIC rhc_c1 = new MIMIC(200, 20, hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\MIMIC\\MIMIC-FourPeaks_" + threshold + ".csv",
                ef_c1, rhc_c1, 2000);
    }

    public static void Train_MIMIC_FlipFlop(AbstractDistribution odd, int[] n)
    {
        DiscreteDependencyTree df = new DiscreteDependencyTree(.1, n);

        FlipFlopEvaluationFunction ef_c1 = new FlipFlopEvaluationFunction();
        GenericProbabilisticOptimizationProblem hcp_c1 = new GenericProbabilisticOptimizationProblem(ef_c1, odd, df);

        MIMIC rhc_c1 = new MIMIC(200, 20, hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\MIMIC\\MIMIC-FlipFlop_.csv",
                ef_c1, rhc_c1, 12000);
    }

    public static void Train_MIMIC_Knapsack(AbstractDistribution odd, int[] n, double[] values, double[] weights, double maximumValue, int[] copiesPerElement)
    {
        DiscreteDependencyTree df = new DiscreteDependencyTree(.1, n);
        KnapsackEvaluationFunction ef_c1 = new KnapsackEvaluationFunction(values, weights, MAX_KNAPSACK_WEIGHT, copiesPerElement);
        GenericProbabilisticOptimizationProblem hcp_c1 = new GenericProbabilisticOptimizationProblem(ef_c1, odd, df);

        MIMIC rhc_c1 = new MIMIC(200, 20, hcp_c1);

        Train("D:\\GT\\MachineLearning\\Randomize\\MIMIC\\MIMIC-Knapsack_.csv",
                ef_c1, rhc_c1, 300);
    }

    private static long mTimeElapsed = 0;
    public static float TrainTest_NN_BP(int t, String outfile, int group, boolean fin, int max_data, int max_iter)
    {
        DataSetReader dsr = new CSVDataSetReader((new File("D:\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm_train_group_" + group + ".csv")).getAbsolutePath());
        DataSet set;
        try {
            set = dsr.read();
        } catch (Exception e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
            return 0.0f;
        }
        Instance ins[] = set.getInstances();
        int len = ins.length;
        if (len > max_data) len = max_data;
        set.setInstances(Arrays.copyOf(ins, len));
        LabelSplitFilter lsf = new LabelSplitFilter();
        lsf.filter(set);
        DiscreteToBinaryFilter dbf = new DiscreteToBinaryFilter();
        dbf.filter(set.getLabelDataSet());
        int outputLayerSize=dbf.getNewAttributeCount();

        BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
        FeedForwardNeuralNetworkFactory factory2 = new FeedForwardNeuralNetworkFactory();
        Vector v = set.get(0).getData();
        System.out.println("Input Layer Size: " + (v.size()-1));
        System.out.println("Output Layer Size: " + outputLayerSize);
        BackPropagationNetwork network = factory.createClassificationNetwork(new int[] { v.size()-1, 5, outputLayerSize }, new LogisticSigmoid());
        FeedForwardNetwork network2 = factory2.createClassificationNetwork(new int[] { v.size()-1, 5, outputLayerSize }, new LogisticSigmoid());

        ConvergenceTrainer trainer = null;
        FixedIterationTrainer trainer2 = null;
        NeuralNetworkOptimizationProblem nno;
        OptimizationAlgorithm o = null;
        switch (t)
        {
            default:
            case 0:
                trainer = new ConvergenceTrainer(new BatchBackPropagationTrainer(set, network, new SumOfSquaresError(), new Adam(0.018, 0.00009, 0.0000999)), 1E-7, max_iter);
                break;
            case 1:
                nno = new NeuralNetworkOptimizationProblem(set, network2, new SumOfSquaresError());
                o = new RandomizedHillClimbing(nno);
                max_iter *= 10;
                trainer2 = new FixedIterationTrainer(o, max_iter);
                //trainer = new ConvergenceTrainer(o, 1E-12, max_iter);
                break;
            case 2:
                nno = new NeuralNetworkOptimizationProblem(set, network2, new SumOfSquaresError());
                o = new SimulatedAnnealing(0.1, 0.00999, nno);
                max_iter *= 10;
                trainer2 = new FixedIterationTrainer(o, max_iter);
                //trainer = new ConvergenceTrainer(o, 1E-7, max_iter);
                break;
            case 3:
                nno = new NeuralNetworkOptimizationProblem(set, network2, new SumOfSquaresError());
                //o = new StandardGeneticAlgorithm(10, 4, 2, nno);
                o = new FitnessOrderedGeneticAlgorithm(20, 10, 5, nno);
                //trainer2 = new FixedIterationTrainer(o, max_iter);
                trainer = new ConvergenceTrainer(o, 5E-11, max_iter);
                break;
        }
        if (t >= 1 && t <= 2 && trainer2 != null)
        {
            long startTime = System.currentTimeMillis();
            trainer2.train();
            mTimeElapsed = System.currentTimeMillis() - startTime;
            Instance opt = o.getOptimal();
            network2.setWeights(opt.getData());
            //network.setWeights(opt.getData());
            System.out.println("Iterations: " + max_iter);
        }
        else
        if (trainer != null) {
            long startTime = System.currentTimeMillis();
            trainer.train();
            mTimeElapsed = System.currentTimeMillis() - startTime;
            System.out.println("Iterations: " + trainer.getIterations());
        }

        DataSetReader dsr_t;
        if (fin)
        {
            dsr_t = new CSVDataSetReader((new File("D:\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm_final_set_" + group + ".csv")).getAbsolutePath());
        }
        else {
            dsr_t = new CSVDataSetReader((new File("D:\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm_test_list_" + group + ".csv")).getAbsolutePath());
            //dsr_t = new CSVDataSetReader((new File("D:\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm_train_group_" + group + ".csv")).getAbsolutePath());
        }
        DataSet set_t;
        try {
            set_t = dsr_t.read();
        } catch (Exception e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
            return 0.0f;
        }
        LabelSplitFilter lsf_t = new LabelSplitFilter();
        lsf_t.filter(set_t);
        DiscreteToBinaryFilter dbf_t = new DiscreteToBinaryFilter();
        dbf_t.filter(set_t.getLabelDataSet());

        float count = 0;
        float matches = 0;
        try {
            FileWriter file = new FileWriter(outfile);
            //"D:\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm_Test_" + group + "_Result.csv"
            //"D:\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm_Final_Result.csv"
            file.write("\n_,VALUE,PRED,CORRECT\n");

            for (int i = 0; i < set_t.size(); i++) {
                Vector result = null;
                if (t == 0) {
                    network.setInputValues(set_t.get(i).getData());
                    network.run();
                    result = network.getOutputValues();
                }
                else
                {
                    network2.setInputValues(set_t.get(i).getData());
                    network2.run();
                    result = network2.getOutputValues();
                }
                //System.out.println("Result " + i + ": " + result);
                Vector target = set_t.get(i).getLabel().getData();
                //System.out.println("Target " + i + ": " + target);
                count = count + 1.0f;
                float r = 0;
                if (result != null) r = Math.round(result.get(0));
                if (r == target.get(0))
                {
                    matches = matches + 1.0f;
                    file.write("_," + target.get(0) + "," + r + ",MATCH\n");
                }
                else
                {
                    file.write("_," + target.get(0) + "," + r + ",FAIL\n");
                }
            }

            if (count != 0) {
                file.write("Tests:," + set_t.size() + ",Matches:," + matches + ",Percent:," + (matches / count));
                System.out.println("Percent: " + (matches / count));
            }
            file.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        if (count == 0.0f) return 0.0f;
        return matches/count;
    }
    
    public static void main(String args[])
    {
        /*
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
        double[] values = new double[NUM_ITEMS];
        double[] weights = new double[NUM_ITEMS];
        for (int i = 0; i < NUM_ITEMS; i++) {
            values[i] = random.nextDouble() * MAX_VALUE;
            weights[i] = random.nextDouble() * MAX_WEIGHT;
        }
        int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);

        int[] n = new int[NUM_ITEMS];
        Arrays.fill(n, 2);
        DiscreteUniformDistribution odd = new DiscreteUniformDistribution(n);
        FixedDistribution fd = new FixedDistribution(odd);
        */

        //Train_HillClimb_FourPeaks(fd, n, 100);
        //Train_HillClimb_FlipFlop(fd, n);
        //Train_HillClimb_Knapsack(fd, n, values, weights, MAX_KNAPSACK_WEIGHT, copies);

        //Train_SimAnneal_FourPeaks(fd, n, 100);
        //Train_SimAnneal_FlipFlop(fd, n);
        //Train_SimAnneal_Knapsack(fd, n, values, weights, MAX_KNAPSACK_WEIGHT, copies);

        //Train_Genetic_FourPeaks(fd, n, 100);
        //Train_Genetic_FlipFlop(fd, n);
        //Train_Genetic_Knapsack(fd, n, values, weights, MAX_KNAPSACK_WEIGHT, copies);

        //Train_MIMIC_FourPeaks(fd, n, 100);
        //Train_MIMIC_FlipFlop(fd, n);
        //Train_MIMIC_Knapsack(fd, n, values, weights, MAX_KNAPSACK_WEIGHT, copies);

        int group = 1;
        int trainer = 3;
        try {
            FileWriter file = new FileWriter("D:\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm" + trainer + "_Result_" + group + "_vsSize.csv");

            for (int i = (500*3); i <= (6000*3); i += (500*3)) {
            //for (int i = (6000*3); i <= (6000*3); i += (500*3)) {
                float result = TrainTest_NN_BP(trainer, "D:\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm" + trainer + "_Test_" + group + "_vsSize" + i + ".csv",
                        group, false, i, 1000);
                file.write("" + i + "," + result + "," + ((float)mTimeElapsed / 1000.0f) + "\n");
            }

            file.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }

        try {
            FileWriter file = new FileWriter("D:\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm" + trainer + "_Result_" + group + "_vsIter.csv");

            for (int i = 5; i <= 300; i += 5) {
                float result = TrainTest_NN_BP(trainer, "D:\\GT\\MachineLearning\\Randomize\\NN2\\NeuralNetwork\\Sarcasm" + trainer + "_Test_" + group + "_vsIter" + i + ".csv",
                        group, false, (6000*3), i);
                file.write("" + i + "," + result + "," + ((float)mTimeElapsed / 1000.0f) + "\n");
            }

            file.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }
}

package opt.test;

import java.io.File;
import java.util.Arrays;
import java.util.ArrayList;
import java.util.List;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.IOException;
import java.io.BufferedReader;
import java.util.Random;
import util.linalg.Vector;

import shared.DataSet;
import shared.DataSetDescription;
import shared.Instance;
import shared.filt.LabelSplitFilter;
import func.KMeansClusterer;
import func.EMClusterer;
import java.util.regex.Pattern;
import shared.filt.PrincipalComponentAnalysis;
import shared.filt.IndependentComponentAnalysis;
import shared.filt.RandomizedProjectionFilter;
import shared.filt.LinearDiscriminantAnalysis;
import util.linalg.Matrix;

import shared.reader.DataSetReader;
import shared.reader.CSVDataSetReader;
import shared.filt.DiscreteToBinaryFilter;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.activation.LogisticSigmoid;
import shared.ConvergenceTrainer;
import shared.FixedIterationTrainer;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.OptimizationAlgorithm;
import func.nn.backprop.BatchBackPropagationTrainer;
import shared.SumOfSquaresError;
import func.nn.backprop.Adam;

import java.util.LinkedList;
import java.util.concurrent.ThreadLocalRandom;
import dist.AbstractConditionalDistribution;
import dist.AbstractConditionalDistribution;
import func.FunctionApproximater;
import shared.DistanceMeasure;
import shared.EuclideanDistance;
import dist.Distribution;
import dist.DiscreteDistribution;

public class Main {

    public static class LinkageClusterer extends AbstractConditionalDistribution implements FunctionApproximater
    {
        public class LinkageGroup
        {
            public LinkageGroup(int n, int max_size, Vector start)
            {
                num = n;
                lst = new int[max_size];
                max = (Vector) start.copy();
                mean = (Vector) start.copy();
                lst[0] = n;
                count = 1;
                used = false;
            }

            public double GetDist(LinkageGroup other)
            {
                return distanceMeasure.value(new Instance(mean), new Instance(other.mean));
            }

            public void Add(LinkageGroup add)
            {
                for (int i = 0; i < add.count; ++i)
                {
                    lst[count++] = add.lst[i];
                }
                max.plusEquals(add.max);
                mean = max.times(1.0/(double)count);
            }

            public boolean used;
            public int num;
            public Vector mean;
            public Vector max;
            public int count;
            public int lst[];
        }

        private int k;
        private DistanceMeasure distanceMeasure;
        LinkedList<LinkageGroup> groups;

        public LinkageClusterer(int k) {
            this.k = k;
            this.distanceMeasure = new EuclideanDistance();
        }

        public void estimate(DataSet set) {
            // Create data
            groups = new LinkedList<LinkageGroup>();
            int size = set.size();
            for (int i = 0; i < size; ++i)
            {
                Instance value = set.get(i);
                groups.add(new LinkageGroup(i, size, value.getData()));
            }
            LinkageGroup rnds[] = new LinkageGroup[200];

            while (size > k) {
                // mark groups as unused
                for (LinkageGroup g : groups) {
                    g.used = false;
                }

                // find 100 random pairs
                int i = 0;
                for (; i < 200; ++i) {
                    int m = size - i;
                    if (m <= 0) break;
                    int randomNum = ThreadLocalRandom.current().nextInt(0, size - i);
                    for (LinkageGroup g : groups) {
                        if (g.used) continue;
                        if (randomNum <= 0) {
                            rnds[i] = g;
                            g.used = true;
                            break;
                        }
                        --randomNum;
                    }
                }

                // find closest distance
                int best = -1;
                double bestDist = 0;
                for (int j = 0; j < i; j += 2)
                {
                    if (j + 1 == i) break;
                    double dst = rnds[j].GetDist(rnds[j+1]);
                    if (best == -1 || dst < bestDist)
                    {
                        best = j;
                        bestDist = dst;
                    }
                }

                // merge them
                int a = best;
                int b = best + 1;
                if (rnds[a].num > rnds[b].num) {
                    ++a;
                    --b;
                }
                if (rnds[a] == rnds[b]) {
                    System.out.print("error");
                }
                rnds[a].Add(rnds[b]);
                groups.remove(rnds[b]);
                --size;
                //System.out.print(":");
                //System.out.flush();
            }

            int k = 0;
            for (LinkageGroup g : groups) {
                g.num = k++;
            }
        }

        public Instance value(Instance data)
        {
            LinkageGroup best = null;
            double bestDist = 0;
            for (LinkageGroup g : groups) {
                double dst = distanceMeasure.value(new Instance(g.mean), data);
                if (best == null || dst < bestDist)
                {
                    best = g;
                    bestDist = dst;
                }

            }
            return new Instance(best.num);
        }

        public Instance[] getClusterCenters()
        {
            Instance result[] = new Instance[k];
            int i = 0;
            for (LinkageGroup g : groups) {
                result[i++] = new Instance(g.mean);
            }
            return result;
        }

        public Distribution distributionFor(Instance instance) {
            double[] distribution = new double[k];
            Instance clusterCenters[] = getClusterCenters();
            for (int i = 0; i < k; i++) {
                distribution[i] +=
                        1/distanceMeasure.value(instance, clusterCenters[i]);
            }
            double sum = 0;
            for (int i = 0; i < distribution.length; i++) {
                sum += distribution[i];
            }
            if (Double.isInfinite(sum)) {
                sum = 0;
                for (int i = 0; i < distribution.length; i++) {
                    if (Double.isInfinite(distribution[i])) {
                        distribution[i] = 1;
                        sum ++;
                    } else {
                        distribution[i] = 0;
                    }
                }
            }
            for (int i = 0; i < distribution.length; i++) {
                distribution[i] /= sum;
            }
            return new DiscreteDistribution(distribution);
        }
    }

    public static DataSet Read(String file, int max_lines) throws Exception {
        BufferedReader br = new BufferedReader(new FileReader(file));
        String line;
        List<Instance> data = new ArrayList<Instance>();
        Pattern pattern = Pattern.compile("[ ,]+");
        int count = 0;
        while ((line = br.readLine()) != null) {
            if (count >= max_lines) break;
            String[] split = pattern.split(line.trim());
            double[] input = new double[split.length];
            for (int i = 0; i < input.length; i++) {
                input[i] = Double.parseDouble(split[i]);
            }
            Instance instance = new Instance(input);
            data.add(instance);
            ++count;
        }
        br.close();
        Instance[] instances = (Instance[]) data.toArray(new Instance[0]);
        DataSet set = new DataSet(instances);
        set.setDescription(new DataSetDescription(set));
        return set;
    }

    public static DataSet LoadData(String file, int max_data, boolean labels)
    {
        DataSet set = null;
        try {
            set = Read(file, max_data);
        } catch (Exception e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
            return null;
        }

        if (labels) {
            LabelSplitFilter lsf = new LabelSplitFilter();
            lsf.filter(set);
        }

        return set;
    }

    public static void k_means(String MyDataSet, String path, String loadfile, String savefile, int maxData)
    {
        DataSet set = LoadData("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\" + path + "\\" + MyDataSet + loadfile + ".csv", maxData, true);

        int pairs[][] = new int[2000][3];
        int pairCount = 0;

        for (int run = 1; run <= 2000; ++run) {
            KMeansClusterer km = new KMeansClusterer();
            km.estimate(set);

            float count = 0;
            float matches = 0;
            int zeroes_v = 0;
            int ones_v = 0;
            int zeroes_o = 0;
            int ones_o = 0;
            try {
                FileWriter file = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\" + path + "\\" + MyDataSet + savefile + run + ".csv");

                for (int i = 0; i < set.size(); ++i) {
                    Instance value = km.value(set.get(i));
                    int v = value.getDiscrete();
                    int o = set.get(i).getLabel().getDiscrete();
                    if (v == 0)
                    {
                        ++zeroes_v;
                    } else
                    {
                        ++ones_v;
                    }
                    if (o == 0) ++zeroes_o; else ++ones_o;
                    if (v == o) {
                        ++matches;
                    }
                    ++count;
                    file.write("" + o + "," + v + "\n");
                }
                if (count > 0.0f) {
                    file.write("" + zeroes_o + "," + zeroes_v + ",:Zeroes\n");
                    file.write("" + ones_o + "," + ones_v + ",:Ones\n");
                    file.write("Tests:," + set.size() + ",Matches:," + matches + ",Percent:," + (matches / count) + "\n");

                    Instance[] centers = km.getClusterCenters();
                    file.write(centers[0].getData().toString() + "\n");
                    file.write(centers[1].getData().toString() + "\n");

                    int s = zeroes_v;
                    int b = ones_v;
                    int t;
                    if (b < s)
                    {
                        t = s;
                        s = b;
                        b = t;
                    }
                    int p = 0;
                    for (; p < pairCount; ++p)
                    {
                        if (s == pairs[p][0])
                        {
                            pairs[p][2] += 1;
                            while (p > 0 && pairs[p][2] > pairs[p-1][2])
                            {
                                t = pairs[p][0];
                                pairs[p][0] = pairs[p-1][0];
                                pairs[p-1][0] = t;

                                t = pairs[p][1];
                                pairs[p][1] = pairs[p-1][1];
                                pairs[p-1][1] = t;

                                t = pairs[p][2];
                                pairs[p][2] = pairs[p-1][2];
                                pairs[p-1][2] = t;

                                --p;
                            }
                            break;
                        }
                    }
                    if (p == pairCount)
                    {
                        pairs[pairCount][0] = s;
                        pairs[pairCount][1] = b;
                        pairs[pairCount][2] = 1;
                        ++pairCount;
                    }

                    System.out.println("% Success: " + (matches / count));
                }
                file.close();
            } catch (Exception e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
                return;
            }
        }

        try {
            FileWriter group = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\" + path + "\\" + MyDataSet + savefile + "_group" + ".csv");

            for (int p = 0; p < pairCount; ++p)
            {
                group.write("" + pairs[p][0] + "," + pairs[p][1] + "," + pairs[p][2] + "\n");
            }

            group.close();
        } catch (Exception e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
            return;
        }
    }

    public static void rand_link(String MyDataSet, String path, String loadfile, String savefile, int maxData)
    {
        DataSet set = LoadData("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\" + path + "\\" + MyDataSet + loadfile + ".csv", maxData, true);
        DataSet lset = set.copy();
        LabelSplitFilter lsf = new LabelSplitFilter();
        lsf.filter(set);

        int pairs[][] = new int[2000][3];
        int pairCount = 0;

        for (int run = 1; run <= 2000; ++run) {
            LinkageClusterer lc = new LinkageClusterer(2);
            lc.estimate(set);

            float count = 0;
            float matches = 0;
            int zeroes_v = 0;
            int ones_v = 0;
            int zeroes_o = 0;
            int ones_o = 0;
            try {
                FileWriter file = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\" + path + "\\" + MyDataSet + savefile + run + ".csv");

                for (int i = 0; i < set.size(); ++i) {
                    Instance value = lc.value(set.get(i));
                    int v = value.getDiscrete();
                    int o = lset.get(i).getLabel().getDiscrete();
                    if (v == 0)
                    {
                        ++zeroes_v;
                    } else
                    {
                        ++ones_v;
                    }
                    if (o == 0) ++zeroes_o; else ++ones_o;
                    if (v == o) {
                        ++matches;
                    }
                    ++count;
                    file.write("" + o + "," + v + "\n");
                }
                if (count > 0.0f) {
                    file.write("" + zeroes_o + "," + zeroes_v + ",:Zeroes\n");
                    file.write("" + ones_o + "," + ones_v + ",:Ones\n");
                    file.write("Tests:," + set.size() + ",Matches:," + matches + ",Percent:," + (matches / count) + "\n");

                    Instance[] centers = lc.getClusterCenters();
                    file.write(centers[0].getData().toString() + "\n");
                    file.write(centers[1].getData().toString() + "\n");

                    int s = zeroes_v;
                    int b = ones_v;
                    int t;
                    if (b < s)
                    {
                        t = s;
                        s = b;
                        b = t;
                    }
                    int p = 0;
                    for (; p < pairCount; ++p)
                    {
                        if (s == pairs[p][0])
                        {
                            pairs[p][2] += 1;
                            while (p > 0 && pairs[p][2] > pairs[p-1][2])
                            {
                                t = pairs[p][0];
                                pairs[p][0] = pairs[p-1][0];
                                pairs[p-1][0] = t;

                                t = pairs[p][1];
                                pairs[p][1] = pairs[p-1][1];
                                pairs[p-1][1] = t;

                                t = pairs[p][2];
                                pairs[p][2] = pairs[p-1][2];
                                pairs[p-1][2] = t;

                                --p;
                            }
                            break;
                        }
                    }
                    if (p == pairCount)
                    {
                        pairs[pairCount][0] = s;
                        pairs[pairCount][1] = b;
                        pairs[pairCount][2] = 1;
                        ++pairCount;
                    }

                    System.out.println("% Success: " + (matches / count));
                }
                file.close();
            } catch (Exception e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
                return;
            }
        }

        try {
            FileWriter group = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\" + path + "\\" + MyDataSet + savefile + "_group" + ".csv");

            for (int p = 0; p < pairCount; ++p)
            {
                group.write("" + pairs[p][0] + "," + pairs[p][1] + "," + pairs[p][2] + "\n");
            }

            group.close();
        } catch (Exception e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
            return;
        }
    }

    public static void exp_max(String MyDataSet, String path, String loadfile, String savefile, int maxData)
    {
        DataSet set = LoadData("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\" + path + "\\" + MyDataSet + loadfile + ".csv", maxData, true);
        DataSet lset = set.copy();
        LabelSplitFilter lsf = new LabelSplitFilter();
        lsf.filter(set);

        for (int run = 1; run < 3; ++run) {
            EMClusterer em = new EMClusterer(2, 0.1, 10);
            em.setDebug(true);
            System.out.println("Estimate start.");
            em.estimate(set);
            System.out.println("Estimate done.");

            float count = 0;
            float matches = 0;
            int zeroes_v = 0;
            int ones_v = 0;
            int zeroes_o = 0;
            int ones_o = 0;
            try {
                FileWriter file = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\" + path + "\\" + MyDataSet + savefile + run + ".csv");
                for (int i = 0; i < set.size(); ++i) {
                    Instance value = em.value(set.get(i));
                    int v = value.getDiscrete();
                    int o = lset.get(i).getLabel().getDiscrete();
                    if (v == 0) ++zeroes_v; else ++ones_v;
                    if (o == 0) ++zeroes_o; else ++ones_o;
                    if (v == o) {
                        ++matches;
                    }
                    ++count;
                    file.write("" + o + "," + v + "\n");
                }
                if (count > 0.0f) {
                    file.write("" + zeroes_o + "," + zeroes_v + ",:Zeroes\n");
                    file.write("" + ones_o + "," + ones_v + ",:Ones\n");
                    file.write("Tests:," + set.size() + ",Matches:," + matches + ",Percent:," + (matches / count));
                    System.out.println("% Success: " + (matches / count));
                }
                file.close();
            } catch (Exception e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
                return;
            }
        }
    }

    public static void pca(String MyDataSet)
    {
        int maxData = 450;
        int group = 1;

        System.out.println("Load.");
        DataSet lset = LoadData("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_train_group_" + group + ".csv", maxData, true);

        for (int run = 1; run < 2; ++run) {
            DataSet set = lset.copy();
            System.out.println("Create filter.");
            PrincipalComponentAnalysis filter = new PrincipalComponentAnalysis(set);
            System.out.println("Start filter.");
            filter.filter(set);
            System.out.println("Finished filter.");
            //try {
            //    FileWriter efile = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_pca_eigen_" + run + ".csv");
            //    Matrix eee = filter.getEigenValues();
            //    for (int z = 0; z < eee.m(); ++z)
            //    {
            //        Vector row = eee.getRow(z);
            //        if (z == eee.m() - 1)
            //        {
            //            efile.write("" + row.get(z));
            //        }
            //        else {
            //            efile.write("" + row.get(z) + ",");
            //        }
            //    }
            //    efile.close();
            //} catch (Exception e) {
            //    System.out.println("An error occurred.");
            //    e.printStackTrace();
            //    return;
            //}
            //if (run == 1) break;
            //Matrix reverse = filter.getProjection().transpose();
            //for (int i = 0; i < set.size(); i++) {
            //    Instance instance = set.get(i);
            //    instance.setData(reverse.times(instance.getData()).plus(filter.getMean()));
            //}
            //System.out.println("Finished reconstruction.");

            float count = 0;
            float matches = 0;
            try {
                FileWriter file = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_pca_result_" + run + ".csv");

                FileWriter mfile = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_pca_strike_" + run + ".csv");
                Instance vv = set.get(0);
                boolean skipList[] = new boolean[vv.size()];
                float th = 1.5f;
                int ccc = 0;
                int mmm = 449; //vv.size() * 3;
                //mmm = mmm >> 2;
                System.out.println("Computing best threshold.");
                do {
                    ccc = 0;
                    for (int j = 0; j < vv.size(); ++j) {
                        skipList[j] = true;
                        for (int k = 0; k < set.size(); ++k) {
                            Instance jv = set.get(k);
                            if (jv.getContinuous(j) > th || jv.getContinuous(j) < -th) {
                                skipList[j] = false;
                                ++ccc;
                                break;
                            }
                        }
                    }
                    th -= 0.01f;
                } while (th > 0 && ccc <= mmm);
                for (int j = 0; j < vv.size(); ++j) {
                    if (skipList[j]) {
                        System.out.println("Skip " + j);
                        mfile.write("" + j + ",");
                    }
                }
                System.out.println("Threashold " + th);
                System.out.println("Count " + ccc); //449
                System.out.println("Max " + vv.size()); //2464
                mfile.close();
                System.out.println("Writing.");

                for (int i = 0; i < lset.size(); ++i) {
                    Instance value = lset.get(i);
                    for (int j = 0; j < value.size(); ++j)
                    {
                        if (!skipList[j]) {
                            file.write("" + value.getDiscrete(j) + ",");
                        }
                    }
                    int v = value.getLabel().getDiscrete();
                    file.write("" + v + "\n");
                }
                file.close();
            } catch (Exception e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
                return;
            }
        }
    }

    public static void ica(String MyDataSet)
    {
        int maxData = 450;
        int group = 1;

        System.out.println("Load.");
        DataSet lset = LoadData("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_train_group_" + group + ".csv", maxData, true);

        for (int run = 1; run < 2; ++run) {
            DataSet set = lset.copy();
            System.out.println("Create filter.");
            IndependentComponentAnalysis filter = new IndependentComponentAnalysis(set);
            System.out.println("Start filter.");
            filter.filter(set);
            System.out.println("Finished filter.");
            //try {
            //    FileWriter efile = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_ica_dist_" + run + ".csv");
            //    Matrix eee = filter.getProjection();
            //    for (int z = 0; z < eee.m(); ++z)
            //    {
            //        Vector row = eee.getRow(z);
            //        if (z == eee.m() - 1)
            //        {
            //            efile.write("" + row.get(z));
            //        }
            //        else {
            //            efile.write("" + row.get(z) + ",");
            //        }
            //    }
            //    efile.close();
            //} catch (Exception e) {
            //    System.out.println("An error occurred.");
            //    e.printStackTrace();
            //    return;
            //}
            //if (run == 1) break;

            float count = 0;
            float matches = 0;
            try {
                FileWriter file = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_ica_result_" + run + ".csv");

                FileWriter mfile = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_ica_strike_" + run + ".csv");
                Instance vv = set.get(0);
                boolean skipList[] = new boolean[vv.size()];
                float th = 1.5f;
                int ccc = 0;
                int mmm = 449; //vv.size() * 3;
                //mmm = mmm >> 2;
                System.out.println("Computing best threshold.");
                do {
                    ccc = 0;
                    for (int j = 0; j < vv.size(); ++j)
                    {
                        skipList[j] = true;
                        for (int k = 0; k < set.size(); ++k)
                        {
                            Instance jv = set.get(k);
                            if (jv.getContinuous(j) > th || jv.getContinuous(j) < -th) { //0.25 - smallest
                                skipList[j] = false;
                                ++ccc;
                                break;
                            }
                        }
                    }
                    th -= 0.01f;
                } while (th > 0 && ccc <= mmm);
                for (int j = 0; j < vv.size(); ++j) {
                    if (skipList[j]) {
                        System.out.println("Skip " + j);
                        mfile.write("" + j + ",");
                    }
                }
                System.out.println("Threashold " + th);
                System.out.println("Count " + ccc);
                System.out.println("Max " + vv.size());
                mfile.close();
                System.out.println("Writing.");

                for (int i = 0; i < lset.size(); ++i) {
                    Instance value = lset.get(i);
                    for (int j = 0; j < value.size(); ++j) {
                        if (!skipList[j]) {
                            file.write("" + value.getDiscrete(j) + ",");
                        }
                    }
                    int v = value.getLabel().getDiscrete();
                    file.write("" + v + "\n");
                }

                file.close();
            } catch (Exception e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
                return;
            }
        }
    }

    public static void rca(String MyDataSet)
    {
        int maxData = 450;
        int group = 1;

        System.out.println("Load.");
        DataSet lset = LoadData("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_train_group_" + group + ".csv", maxData, true);

        for (int run = 1; run < 50; ++run) {
            DataSet set = lset.copy();
            System.out.println("Create filter.");
            Instance ww = set.get(0);
            RandomizedProjectionFilter filter = new RandomizedProjectionFilter(ww.size(), ww.size());
            System.out.println("Start filter.");
            filter.filter(set);
            System.out.println("Finished filter.");

            float count = 0;
            float matches = 0;
            try {
                FileWriter file = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_rca_result_" + run + ".csv");

                FileWriter mfile = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_rca_strike_" + run + ".csv");
                Instance vv = set.get(0);
                boolean skipList[] = new boolean[vv.size()];
                float th = 1.5f;
                int ccc = 0;
                int mmm = 449; //vv.size() * 3;
                //mmm = mmm >> 2;
                System.out.println("Computing best threshold.");
                do {
                    ccc = 0;
                    for (int j = 0; j < vv.size(); ++j)
                    {
                        skipList[j] = true;
                        for (int k = 0; k < set.size(); ++k)
                        {
                            Instance jv = set.get(k);
                            if (jv.getContinuous(j) > th || jv.getContinuous(j) < -th) {
                                skipList[j] = false;
                                ++ccc;
                                break;
                            }
                        }
                    }
                    th -= 0.01f;
                } while (th > 0 && ccc <= mmm);
                for (int j = 0; j < vv.size(); ++j) {
                    if (skipList[j]) {
                        System.out.println("Skip " + j);
                        mfile.write("" + j + ",");
                    }
                }
                System.out.println("Threashold " + th);
                System.out.println("Count " + ccc);
                System.out.println("Max " + vv.size());
                mfile.close();
                System.out.println("Writing.");

                for (int i = 0; i < lset.size(); ++i) {
                    Instance value = lset.get(i);
                    for (int j = 0; j < value.size(); ++j)
                    {
                        if (!skipList[j]) {
                            file.write("" + value.getDiscrete(j) + ",");
                        }
                    }
                    int v = value.getLabel().getDiscrete();
                    file.write("" + v + "\n");
                }
                file.close();
            } catch (Exception e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
                return;
            }
        }
    }

    public static void lca(String MyDataSet)
    {
        int maxData = 450;
        int group = 1;

        System.out.println("Load.");
        DataSet lset = LoadData("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_train_group_" + group + ".csv", maxData, true);

        for (int run = 1; run < 2; ++run) {
            DataSet set = lset.copy();
            LabelSplitFilter lsf = new LabelSplitFilter();
            lsf.filter(set);
            System.out.println("Create filter.");
            LinearDiscriminantAnalysis filter = new LinearDiscriminantAnalysis(set);
            System.out.println("Start filter.");
            filter.filter(set);
            System.out.println("Finished filter.");
            try {
                FileWriter efile = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_lca_dist_" + run + ".csv");
                for (int z = 0; z < set.size(); ++z)
                {
                    Instance row = set.get(z);
                    if (z == set.size() - 1)
                    {
                        efile.write("" + row.getContinuous(0));
                    }
                    else {
                        efile.write("" + row.getContinuous(0) + ",");
                    }
                }
                efile.close();
            } catch (Exception e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
                return;
            }
            if (run == 1) break;
            filter.reverse(set);
            System.out.println("Finished reconstruction.");

            float count = 0;
            float matches = 0;
            try {
                FileWriter file = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_lca_result_" + run + ".csv");

                FileWriter mfile = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_lca_strike_" + run + ".csv");
                Instance vv = set.get(0);
                boolean skipList[] = new boolean[vv.size()];
                float th = 5.0f;
                int ccc = 0;
                int mmm = 449; //vv.size() * 3;
                //mmm = mmm >> 2;
                System.out.println("Computing best threshold.");
                do {
                    ccc = 0;
                    for (int j = 0; j < vv.size(); ++j)
                    {
                        skipList[j] = true;
                        for (int k = 0; k < set.size(); ++k)
                        {
                            Instance jv = set.get(k);
                            if (jv.getContinuous(j) > th || jv.getContinuous(j) < -th) {
                                skipList[j] = false;
                                ++ccc;
                                break;
                            }
                        }
                    }
                    th -= 0.01f;
                } while (th > 0 && ccc <= mmm);
                for (int j = 0; j < vv.size(); ++j) {
                    if (skipList[j]) {
                        System.out.println("Skip " + j);
                        mfile.write("" + j + ",");
                    }
                }
                System.out.println("Threashold " + th);
                System.out.println("Count " + ccc);
                System.out.println("Max " + vv.size());
                mfile.close();
                System.out.println("Writing.");

                for (int i = 0; i < lset.size(); ++i) {
                    Instance value = lset.get(i);
                    for (int j = 0; j < value.size() - 1; ++j)
                    {
                        if (!skipList[j]) {
                            file.write("" + value.getDiscrete(j) + ",");
                        }
                    }
                    int v = value.getLabel().getDiscrete();
                    file.write("" + v + "\n");
                }
                file.close();
            } catch (Exception e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
                return;
            }
        }
    }

    private static long mIterations = 0;
    private static long mTimeElapsed = 0;
    public static float TrainTest_NN_BP(String infile, String testfile, String strikefile, String tempfile, String outfile, int max_data, int max_iter)
    {
        DataSetReader dsr = new CSVDataSetReader((new File(infile)).getAbsolutePath());
        DataSet set;
        try {
            set = dsr.read();
        } catch (Exception e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
            return 0.0f;
        }
        System.out.println("Data read.");
        Instance ins[] = set.getInstances();
        int len = ins.length;
        if (len > max_data) len = max_data;
        set.setInstances(Arrays.copyOf(ins, len));
        LabelSplitFilter lsf = new LabelSplitFilter();
        lsf.filter(set);
        System.out.println("Filtered.");
        DiscreteToBinaryFilter dbf = new DiscreteToBinaryFilter();
        dbf.filter(set.getLabelDataSet());
        int outputLayerSize=dbf.getNewAttributeCount();

        BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
        Vector v = set.get(0).getData();
        System.out.println("Input Layer Size: " + (v.size()-1));
        System.out.println("Output Layer Size: " + outputLayerSize);
        BackPropagationNetwork network = factory.createClassificationNetwork(new int[] { v.size()-1, 5, outputLayerSize }, new LogisticSigmoid());

        ConvergenceTrainer trainer = null;
        FixedIterationTrainer trainer2 = null;
        NeuralNetworkOptimizationProblem nno;
        OptimizationAlgorithm o = null;
        trainer = new ConvergenceTrainer(new BatchBackPropagationTrainer(set, network, new SumOfSquaresError(), new Adam(0.018, 0.00009, 0.0000999)), 1E-7, max_iter);
        if (trainer != null) {
            long startTime = System.currentTimeMillis();
            trainer.train();
            mTimeElapsed = System.currentTimeMillis() - startTime;
            System.out.println("Iterations: " + trainer.getIterations());
            mIterations = trainer.getIterations();
        }

        DataSetReader dsr_tt;
        dsr_tt = new CSVDataSetReader((new File(testfile)).getAbsolutePath());
        DataSet set_tt;
        try {
            set_tt = dsr_tt.read();
        } catch (Exception e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
            return 0.0f;
        }
        LabelSplitFilter lsf_tt = new LabelSplitFilter();
        lsf_tt.filter(set_tt);
        DiscreteToBinaryFilter dbf_tt = new DiscreteToBinaryFilter();
        dbf_tt.filter(set_tt.getLabelDataSet());

        DataSetReader dsr_s = null;
        DataSet set_s = null;
        if (strikefile != "") {
            dsr_s = new CSVDataSetReader((new File(strikefile)).getAbsolutePath());
            try {
                set_s = dsr_s.read();
            } catch (Exception e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
                return 0.0f;
            }
        }
        try {
            FileWriter tmp = new FileWriter(tempfile);
            Instance strike = null;
            if (dsr_s != null) strike = set_s.get(0);
            int si = 0;
            for (int i = 0; i < set_tt.size(); i++) {
                Instance value = set_tt.get(i);
                si = 0;
                for (int j = 0; j < value.size(); ++j) {
                    if (strike != null && si < strike.size()) {
                        int d;
                        do {
                            d = strike.getDiscrete(si);
                            if (d < j) ++si;
                        } while (d < j && si < strike.size());
                        if (d == j) continue;
                    }
                    tmp.write("" + value.getDiscrete(j) + ",");
                }
                int vvv = value.getLabel().getDiscrete();
                tmp.write("" + vvv + "\n");
            }
            tmp.close();
        } catch (Exception e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
            return 0.0f;
        }

        DataSetReader dsr_t = new CSVDataSetReader((new File(tempfile)).getAbsolutePath());
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
            file.write("\n_,VALUE,PRED,CORRECT\n");

            for (int i = 0; i < set_t.size(); i++) {
                System.out.flush();
                Vector result = null;
                network.setInputValues(set_t.get(i).getData());
                network.run();
                result = network.getOutputValues();
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

    public static void Clusters(String MyDataSet)
    {
        //k_means(MyDataSet,"data_small", "_train_group_1", "_kmeans_result_", 450);
        //exp_max(MyDataSet, "data_smallest", "_train_group_1", "_expmax_result_", 150);
        rand_link(MyDataSet,"data_small", "_train_group_1", "_rlink_result_", 450);
    }

    public static void Reductions(String MyDataSet)
    {
        //pca(MyDataSet);
        //ica(MyDataSet);
        //rca(MyDataSet);
        lca(MyDataSet);
    }

    public static void ReductionsClusters(String MyDataSet)
    {
        int trial = 1;
        //k_means(MyDataSet,"data_small", "_pca_result_" + trial, "_pca" + trial + "_kmeans_result_", 450);
        //exp_max(MyDataSet, "data_small", "_pca_result_" + trial, "_pca" + trial + "_expmax_result_", 450);
        //rand_link(MyDataSet,"data_small", "_pca_result_" + trial, "_pca" + trial + "_rlink_result_", 450);
        //k_means(MyDataSet,"data_small", "_ica_result_" + trial, "_ica" + trial + "_kmeans_result_", 450);
        //rand_link(MyDataSet,"data_small", "_ica_result_" + trial, "_ica" + trial + "_rlink_result_", 450);
        //for (int i = 1; i < 50; ++i) k_means(MyDataSet,"data_small", "_rca_result_" + i, "_rca" + i + "_kmeans_result_", 450);
        //for (int i = 1; i < 50; ++i) rand_link(MyDataSet,"data_small", "_rca_result_" + i, "_rca" + i + "_rlink_result_", 450);
        k_means(MyDataSet,"data_small", "_lca_result_" + trial, "_lca" + trial + "_kmeans_result_", 450);
        rand_link(MyDataSet,"data_small", "_lca_result_" + trial, "_lca" + trial + "_rlink_result_", 450);
    }

    public static void NN_Reductions(String MyDataSet, String red, int trial)
    {
        try {
            FileWriter file = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn\\" + MyDataSet + "_" + red + "_nn_iterations.csv");

            for (int i = 1000; i <= 1000; i += 5) {
                float result = TrainTest_NN_BP("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_" + red + "_result_" + trial + ".csv",
                        "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_test_list_1.csv",
                        "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\data_small\\" + MyDataSet + "_" + red + "_strike_" + trial + ".csv",
                        "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn\\temp.csv",
                        "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn\\" + MyDataSet + "_" + red + "_nn_result_" + trial + ".csv",
                        450, i);
                file.write("" + mIterations + "," + result + "," + ((float)mTimeElapsed / 1000.0f) + "\n");
            }

            file.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    public static void NN_ReductionsClusters(String MyDataSet, String red, String clstr, int trial)
    {
        try {
            FileWriter file = new FileWriter("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2\\" + MyDataSet + "_" + red + "_" + clstr + "_nn_iterations.csv");

            for (int i = 1000; i <= 1000; i += 5) {
                float result = TrainTest_NN_BP("D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2\\" + MyDataSet + "_train_" + red + trial + "_" + clstr + "_result.csv",
                        "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2\\" + MyDataSet + "_test_" + red + trial + "_" + clstr + "_result.csv",
                        "",
                        "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2\\temp.csv",
                        "D:\\School\\GT\\MachineLearning\\UnsupervisedLearning\\NN\\nn2\\" + MyDataSet + "_" + red + "_" + clstr + "_nn_result_" + trial + ".csv",
                        450, i);
                file.write("" + mIterations + "," + result + "," + ((float)mTimeElapsed / 1000.0f) + "\n");
            }

            file.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }

    public static void main(String args[])
    {
        String MyDataSet = "Sarcasm";
        //Clusters(MyDataSet);
        //Reductions(MyDataSet);
        //ReductionsClusters(MyDataSet);

        MyDataSet = "Sentiment";
        //Clusters(MyDataSet);
        //Reductions(MyDataSet);
        //ReductionsClusters(MyDataSet);

        MyDataSet = "Sarcasm";
        //NN_Reductions(MyDataSet, "pca", 1);
        //NN_Reductions(MyDataSet, "ica", 1);
        //for (int i = 33; i < 50; ++i) NN_Reductions(MyDataSet, "rca", i);
        NN_Reductions(MyDataSet, "lca", 1);

        //NN_ReductionsClusters(MyDataSet, "pca", "kmeans", 1);
        //NN_ReductionsClusters(MyDataSet, "ica", "kmeans", 1);
        //NN_ReductionsClusters(MyDataSet, "rca", "kmeans", 1);
        NN_ReductionsClusters(MyDataSet, "lca", "kmeans", 1);
        //NN_ReductionsClusters(MyDataSet, "pca", "rlink", 1);
        //NN_ReductionsClusters(MyDataSet, "ica", "rlink", 1);
        //NN_ReductionsClusters(MyDataSet, "rca", "rlink", 1);
        NN_ReductionsClusters(MyDataSet, "lca", "rlink", 1);
    }
}

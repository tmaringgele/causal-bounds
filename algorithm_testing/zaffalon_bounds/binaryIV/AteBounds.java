package binaryIV;

import ch.idsia.credici.inference.CausalMultiVE;
import ch.idsia.credici.inference.CausalVE;
import ch.idsia.credici.inference.CredalCausalVE;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.credici.model.builder.CausalBuilder;
import ch.idsia.credici.utility.DAGUtil;
import ch.idsia.credici.utility.DataUtil;
import ch.idsia.credici.utility.FactorUtil;
import ch.idsia.credici.utility.experiments.Logger;
import ch.idsia.crema.factor.bayesian.BayesianFactor;
import ch.idsia.crema.factor.credal.vertex.VertexFactor;
import ch.idsia.crema.model.graphical.SparseDirectedAcyclicGraph;
import ch.idsia.crema.utility.RandomUtil;
import ch.idsia.credici.model.builder.EMCredalBuilder;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;
import ch.idsia.credici.inference.CausalInference;

import java.io.File;
import java.io.Reader;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.io.FileWriter;
import java.io.PrintWriter;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class AteBounds {
    public static void main(String[] args) throws Exception {

        int b_X_Y_1000 = -5000;
        try (PrintWriter writer = new PrintWriter(new FileWriter("zaffalon_bounds/binaryIV/results.csv"))) {
            writer.println("b_X_Y_1000,zaffalon_bound_lower,zaffalon_bound_upper");
            /*
             * Running 2000 Iterations of this will take 4 hours on my laptop.
             * It will take 10minutes on a google c4-highcpu-32
             */
            for (int i = 0; i < 4; i++) {
                // Variable IDs for clarity
                int Z = 0, X = 1, Y = 2, U = 3;

                // Load data from CSV file
                TIntIntMap[] data = getDataFromCSV("zaffalon_bounds/binaryIV/data_for_zaffalon.csv", Z, X, Y, b_X_Y_1000);

                double[] bounds = getBoundsForBinaryIV(data, 100, 30, "ate", Z, X, Y, U);

                // Print the bounded results
                System.out.printf("Result number %d: for b_X_Y_1000 = %d%n", i, b_X_Y_1000);
                System.out.printf("ATE (ACE) bounds: [%.4f, %.4f]%n", bounds[0], bounds[1]);

                // Save results to CSV
                writer.printf("%d,%.4f,%.4f%n", b_X_Y_1000, bounds[0], bounds[1]);

                b_X_Y_1000 += 5;
            }
        }
    }

    public static TIntIntMap[] getDataFromCSV(String filePath, int Z, int X, int Y) throws Exception {
        List<TIntIntMap> dataList = new ArrayList<>();
        try (Reader reader = Files.newBufferedReader(Paths.get("zaffalon_bounds/repo/examples/data300.csv"));
                CSVParser csv = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
            for (CSVRecord record : csv) {
                int zVal = Integer.parseInt(record.get("Z"));
                int xVal = Integer.parseInt(record.get("X"));
                int yVal = Integer.parseInt(record.get("Y"));
                // Create a map of variable -> value for this sample
                TIntIntMap sample = new TIntIntHashMap();
                sample.put(Z, zVal);
                sample.put(X, xVal);
                sample.put(Y, yVal);
                dataList.add(sample);
            }
        }
        TIntIntMap[] data = dataList.toArray(new TIntIntMap[0]);
        // // Alternative - using the credici library.
        // TIntIntMap[] data = DataUtil.fromCSV("zaffalon_bounds/repo/examples/data300_idx.csv");
        return data;
    }

    public static TIntIntMap[] getDataFromCSV(String filePath, int Z, int X, int Y, int b_X_Y_1000) throws Exception {
        List<TIntIntMap> dataList = new ArrayList<>();
        try (Reader reader = Files.newBufferedReader(Paths.get(filePath));
                CSVParser csv = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
            for (CSVRecord record : csv) {
                int bValue = Integer.parseInt(record.get("b_X_Y_1000"));
                if (bValue == b_X_Y_1000) {
                    int zVal = Integer.parseInt(record.get("Z"));
                    int xVal = Integer.parseInt(record.get("X"));
                    int yVal = Integer.parseInt(record.get("Y"));
                    // Create a map of variable -> value for this sample
                    TIntIntMap sample = new TIntIntHashMap();
                    sample.put(Z, zVal);
                    sample.put(X, xVal);
                    sample.put(Y, yVal);
                    dataList.add(sample);
                }
            }
        }
        TIntIntMap[] data = dataList.toArray(new TIntIntMap[0]);
        return data;
    }

    public static double[] getBoundsForBinaryIV(TIntIntMap[] data , int maxIter, int runs, String query, int Z, int X, int Y, int U) throws Exception {


        

        // Endogenous DAG (only X -> Y among observed endogenous variables)
        SparseDirectedAcyclicGraph endoDAG = DAGUtil.build("(1,2)");

        // Complete causal DAG including exogenous influences (Z->X, U->X, U->Y, X->Y)
        SparseDirectedAcyclicGraph causalDAG = DAGUtil.build("(0,1),(3,1),(3,2),(1,2)");

        // Build the SCM with binary variables (cardinality 2) and the specified DAG
        StructuralCausalModel scm = CausalBuilder.of(causalDAG, 2)
                .build();
        // scm.fillExogenousWithRandomFactors(42); // initialize exogenous Z and U with random PMFs

        

        // Run EM to learn a set of compatible SCM parameterizations
        EMCredalBuilder builder = EMCredalBuilder.of(scm, data)
                .setMaxEMIter(maxIter)
                .setNumTrajectories(runs)
                .setWeightedEM(true) // use weighted EM (for convergence stability)
                .build();

        // Retrieve the set of learned SCMs (extreme points of the credal set)
        List<StructuralCausalModel> modelSet = builder.getSelectedPoints();

        // Initialize multi-model inference over the set of learned SCMs
        CausalMultiVE multiVE = new CausalMultiVE(modelSet);


        if(query.equals("pns")){
        // Query 1: Probability of Necessity and Sufficiency for X->Y
        VertexFactor pnsFactor = (VertexFactor) multiVE.probNecessityAndSufficiency(X, Y);
        // Extract lower and upper values for PNS from the VertexFactor
        double pnsLower = pnsFactor.getData()[0][0][0];
        double pnsUpper = pnsFactor.getData()[0][1][0];
        double[] pnsBounds = new double[] { pnsLower, pnsUpper };
        return pnsBounds;
        } else if(query.equals("ate")){

        // Query 2: Bounds on ATE = P(Y|do X=1) – P(Y|do X=0)
        VertexFactor ace = (VertexFactor) multiVE.averageCausalEffects(X, Y, 1, 1, 0);

        

        double aceLower = ace.getData()[0][0][0]; // lower bound
        double aceUpper = ace.getData()[0][1][0]; // upper bound

        double[] bounds = new double[] { aceLower, aceUpper };
        return bounds;
        }
        else {
            throw new IllegalArgumentException("Invalid query type. Use 'pns' or 'ate'.");
        }
    }
    
}

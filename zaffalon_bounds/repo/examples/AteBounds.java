package repo.examples;

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

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

public class AteBounds {
    public static void main(String[] args) throws Exception {
        // Variable IDs for clarity
        int Z = 0, X = 1, Y = 2, U = 3;  

        // Endogenous DAG (only X -> Y among observed endogenous variables)
        SparseDirectedAcyclicGraph endoDAG = DAGUtil.build("(1,2)");  

        // Complete causal DAG including exogenous influences (Z->X, U->X, U->Y, X->Y)
        SparseDirectedAcyclicGraph causalDAG = DAGUtil.build("(0,1),(3,1),(3,2),(1,2)");
        

        // Build the SCM with binary variables (cardinality 2) and the specified DAG
        StructuralCausalModel scm = CausalBuilder.of(causalDAG, 2)
        .build();
        // scm.fillExogenousWithRandomFactors(42); // initialize exogenous Z and U with random PMFs

        
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
        // TIntIntMap[] data = dataList.toArray(new TIntIntMap[0]);

        TIntIntMap[] data = DataUtil.fromCSV("zaffalon_bounds/repo/examples/data300_idx.csv");


        int maxIter = 100;        // maximum EM iterations per run
        int runs = 10;            // number of random restarts (trajectories)

        // Run EM to learn a set of compatible SCM parameterizations
        EMCredalBuilder builder = EMCredalBuilder.of(scm, data)
            .setMaxEMIter(maxIter)
            .setNumTrajectories(runs)
            .setWeightedEM(false)      // use weighted EM (for convergence stability)
            .build();

        // Retrieve the set of learned SCMs (extreme points of the credal set)
        List<StructuralCausalModel> modelSet = builder.getSelectedPoints();

        System.out.println("Learned " + modelSet.size() + " models.");
        System.out.println("first model: ");
        modelSet.get(0).printSummary();

        // Initialize multi-model inference over the set of learned SCMs
        CausalMultiVE multiVE = new CausalMultiVE(modelSet);

        // Query 1: Probability of Necessity and Sufficiency for X->Y
        VertexFactor pnsFactor = (VertexFactor) multiVE.probNecessityAndSufficiency(X, Y);

        // Query 2: Bounds on ATE = P(Y|do X=1) – P(Y|do X=0)
        VertexFactor ace = (VertexFactor) multiVE.averageCausalEffects(X, Y, 1, 1, 0);

        double aceLower = ace.getData()[0][0][0];  // lower bound
        double aceUpper = ace.getData()[0][1][0];  // upper bound


        // Extract lower and upper values for PNS from the VertexFactor
        double pnsLower = pnsFactor.getData()[0][0][0];
        double pnsUpper = pnsFactor.getData()[0][1][0];

        System.out.println("Maximum EM iterations: " + maxIter);
        System.out.println("Number of random restarts: " + runs);

        // Print the bounded results
        System.out.printf("ATE (ACE) bounds: [%.4f, %.4f]%n", aceLower, aceUpper);
        System.out.printf("PNS bounds: [%.4f, %.4f]%n", pnsLower, pnsUpper);
        


    }
    
}

package repo.examples;

import ch.idsia.credici.inference.CausalMultiVE;
import ch.idsia.credici.inference.CausalVE;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.credici.model.builder.CausalBuilder;
import ch.idsia.credici.utility.DAGUtil;
import ch.idsia.credici.utility.DataUtil;
import ch.idsia.credici.utility.FactorUtil;
import ch.idsia.credici.utility.experiments.Logger;
import ch.idsia.crema.factor.credal.vertex.VertexFactor;
import ch.idsia.crema.model.graphical.SparseDirectedAcyclicGraph;
import ch.idsia.crema.utility.RandomUtil;
import ch.idsia.credici.model.builder.EMCredalBuilder;
import gnu.trove.map.TIntIntMap;
import ch.idsia.credici.inference.CausalInference;


import java.io.File;
import java.util.HashMap;
import java.util.List;

public class InstrumentalPNSExample {
    public static void main(String[] args) throws Exception {
        Logger logger = new Logger();
        RandomUtil.setRandomSeed(0);

        // Variable IDs
        int Z = 0, X = 1, Y = 2;

        // Define DAG with only observed variables: Z → X → Y
        SparseDirectedAcyclicGraph dag = DAGUtil.build("(0,1),(1,2)");

        // Build SCM without unobserved confounder
        StructuralCausalModel model = CausalBuilder.of(dag, 2).build();

        logger.info("Model structure: " + model.getNetwork());

        // Load observed CSV data (Z,X,Y)
        File csvFile = new File("zaffalon_bounds/repo/examples/data.csv");
        TIntIntMap[] data = CSVReader.readCSV(csvFile); // expects 0/1 values, no header

        logger.info("Loaded data: " + data.length + " samples");

        // Compute empirical distribution
        HashMap empiricalDist = DataUtil.getEmpiricalMap(model, data);
        empiricalDist = FactorUtil.fixEmpiricalMap(empiricalDist, 6);

        // Learn a set of compatible models via EMCC
        EMCredalBuilder builder = EMCredalBuilder.of(model, data)
                .setMaxEMIter(50)
                .setNumTrajectories(5)
                .setWeightedEM(true)
                .build();

        // Inference: compute bounds on PNS (probability of necessity & sufficiency)
        CausalMultiVE multiVE = new CausalMultiVE(builder.getSelectedPoints());
        VertexFactor pns = (VertexFactor) multiVE.probNecessityAndSufficiency(X, Y);

        VertexFactor ace = (VertexFactor) multiVE.averageCausalEffects(X, Y);

        double ace_lower = ace.getData()[0][0][0];
        double ace_upper = ace.getData()[0][1][0];

        double lower = pns.getData()[0][0][0];
        double upper = pns.getData()[0][1][0];

        System.out.printf("ACE (=ATE) bounds: [%.4f, %.4f]%n", ace_lower, ace_upper);

        System.out.printf("PNS bounds: [%.4f, %.4f]%n", lower, upper);
    }
}

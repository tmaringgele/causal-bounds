package repo.examples;

import ch.idsia.credici.IO;
import ch.idsia.credici.inference.CausalMultiVE;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.credici.model.builder.CausalBuilder;
import ch.idsia.credici.utility.DAGUtil;
import ch.idsia.credici.utility.DataUtil;
import ch.idsia.credici.utility.FactorUtil;
import ch.idsia.credici.utility.Probability;
import ch.idsia.credici.utility.apps.SelectionBias;
import ch.idsia.credici.utility.experiments.Logger;
import ch.idsia.credici.utility.experiments.Watch;
import ch.idsia.crema.factor.credal.vertex.VertexFactor;
import ch.idsia.crema.model.graphical.SparseDirectedAcyclicGraph;
import ch.idsia.crema.utility.ArraysUtil;
import ch.idsia.crema.utility.RandomUtil;
import com.opencsv.exceptions.CsvException;
import gnu.trove.map.TIntIntMap;
import jdk.jshell.spi.ExecutionControl;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;


public class AteBounds_manual {

    public static void main(String[] args) throws InterruptedException, ExecutionControl.NotImplementedException, IOException, CsvException {
        int maxIter = 500;
        int executions = 10;

        Logger logger = new Logger();
        String prjPath = "zaffalon_bounds/repo/";
        String wdir = Path.of(prjPath).toString();

        int Z = 0;
        int X = 1;
        int Y = 2;
        int U = 3;
        int W = 4;

        // Define empirical dag (only observed variables) (Z->X, X->Y)
        // SparseDirectedAcyclicGraph endoDag = DAGUtil.build("(0,1),(1,2)");
        SparseDirectedAcyclicGraph endoDag = DAGUtil.build("(0,1)");

        // Define complete causal DAG (Z->X, U->X, W->Y, X->Y)
        // SparseDirectedAcyclicGraph causalDAG = DAGUtil.build("(0,1),(3,1),(4,2),(1,2)");
        SparseDirectedAcyclicGraph causalDAG = DAGUtil.build("(0,1),(2,0),(3,1)");


        // Build SCM from Dag, with endogenous variable size 2 (binary variables)
        // Has to contain headline with variables encoded as IDs: ',0,1,2' for ',Z,X,Y'

        int[] exoVarSizes = new int[]{4, 4, 4}; // cardinality of the exogenous variables (Z=4, U=4)

        StructuralCausalModel scm = CausalBuilder.of(endoDag, 2)
                // .setExoVarSizes(exoVarSizes)
                .setCausalDAG(causalDAG)
                .build();

        // load simulation data with indexes
        TIntIntMap[] data = DataUtil.fromCSV(wdir + "/examples/data300_idx_noZ.csv");
        
        // Empirical endogenous distribution from the data
        HashMap empiricalDist = DataUtil.getEmpiricalMap(scm, data);
        empiricalDist = FactorUtil.fixEmpiricalMap(empiricalDist, 6);

        logger.info("Sampled complete data with size: " + data.length);
        
        int[][] hidden_conf = new int[][]{{0,0},{1,1}};
        int[] Sparents = new int[] { X, Y };

        // Model with Selection Bias structure
        StructuralCausalModel modelBiased = SelectionBias.addSelector(scm, Sparents, hidden_conf);
        int selectVar = ArraysUtil.difference(modelBiased.getEndogenousVars(), scm.getEndogenousVars())[0];

        // Biased data
        TIntIntMap[] dataBiased = SelectionBias.applySelector(data, modelBiased, selectVar);

        logger.info("Running EMCC with selected data.");

        // Learn the model
        List endingPoints = SelectionBias.runEM(modelBiased, selectVar, dataBiased, maxIter, executions);

        // Run inference
        CausalMultiVE multiInf = new CausalMultiVE(endingPoints);
        VertexFactor p = (VertexFactor) multiInf.probNecessityAndSufficiency(X, Y);
        logger.info("PNS: ["+p.getData()[0][0][0]+", "+p.getData()[0][1][0]+"]");


    }

}

package repo.binaryIV;
import ch.idsia.credici.IO;
import ch.idsia.credici.factor.EquationBuilder;
import ch.idsia.credici.inference.CausalMultiVE;
import ch.idsia.credici.inference.CredalCausalVE;
import ch.idsia.credici.model.StructuralCausalModel;
import ch.idsia.credici.model.builder.CausalBuilder;
import ch.idsia.credici.model.builder.EMCredalBuilder;
import ch.idsia.credici.utility.DataUtil;
import ch.idsia.credici.utility.FactorUtil;
import ch.idsia.crema.factor.bayesian.BayesianFactor;
import ch.idsia.crema.factor.credal.linear.SeparateHalfspaceFactor;
import ch.idsia.crema.factor.credal.vertex.VertexFactor;
import ch.idsia.crema.model.graphical.SparseModel;
import ch.idsia.crema.model.graphical.specialized.BayesianNetwork;
import gnu.trove.map.TIntIntMap;
import jdk.jshell.spi.ExecutionControl;
import org.junit.Assert;

import java.io.IOException;
import java.nio.file.Path;
import java.util.HashMap;


public class generateBinaryIVModel {
    public static void main(String[] args) throws Exception {

        // 1. Setup
        String wdir = ".";
        String folder = Path.of(wdir, "zaffalon_bounds/repo/binaryIV").toString();

        // 2. Define the Bayesian network
        BayesianNetwork bnet = new BayesianNetwork();

        int Z = bnet.addVariable(2); // Instrument
        int X = bnet.addVariable(2); // Treatment
        int Y = bnet.addVariable(2); // Outcome

        bnet.addParents(X, Z); // Z → X
        bnet.addParents(Y, X); // X → Y

        // 3. Variable names
        HashMap<Integer, String> varNames = new HashMap<>();
        varNames.put(Z, "Z");
        varNames.put(X, "X");
        varNames.put(Y, "Y");

        int z = 1, x = 1, y = 1;
        int z_ = 0, x_ = 0, y_ = 0;

        // 4. Build empirical counts
        BayesianFactor counts = new BayesianFactor(bnet.getDomain(Z, X, Y));

        // Z = 0
        counts.setValue(50, z_, x_, y_);
        counts.setValue(30, z_, x_, y);
        counts.setValue(10, z_, x, y_);
        counts.setValue(60, z_, x, y);

        // Z = 1
        counts.setValue(15, z, x_, y_);
        counts.setValue(25, z, x_, y);
        counts.setValue(65, z, x, y_);
        counts.setValue(95, z, x, y);

        // 5. Sanity check
        FactorUtil.print(counts.reorderDomain(Y, X, Z), varNames);
        int N = (int) counts.marginalize(X, Y, Z).getValueAt(0);
        System.out.println("Total samples: " + N);

        // 6. Compute marginals and conditionals
        BayesianFactor nz = counts.marginalize(X, Y);     // P(Z)
        BayesianFactor nxz = counts.marginalize(Y);       // P(X, Z)
        BayesianFactor nxy = counts.marginalize(Z);       // P(X, Y)

        BayesianFactor px_z = nxz.divide(nz);             // P(X | Z)
        BayesianFactor py_x = nxy.divide(nxy.marginalize(Y)); // P(Y | X)
        BayesianFactor pz = nz.scalarMultiply(1.0 / N);   // P(Z)

        // 7. Create dataset from counts
        TIntIntMap[] data = DataUtil.dataFromCounts(counts);

        // 8. Build Bayesian network
        bnet.setFactor(Z, pz);
        bnet.setFactor(X, px_z);
        bnet.setFactor(Y, py_x);

        // 9. Build SCM
        StructuralCausalModel scm = CausalBuilder.of(bnet).build();

        // Optional: deterministic function X = f(Z)
        // BayesianFactor fx = EquationBuilder.of(scm).fromVector(X, 0, 0, 1, 1);
        // scm.setFactor(X, fx);

        // Fill latent variables randomly
        scm.fillExogenousWithRandomFactors(3);

        // 10. Empirical distribution from generated data
        HashMap empiricalDist = DataUtil.getEmpiricalMap(scm, data);
        empiricalDist = FactorUtil.fixEmpiricalMap(empiricalDist, 6);

        System.out.println("Empirical distribution:");
        System.out.println(empiricalDist);

        // 11. Save model and data
        IO.write(scm, folder + "/iv_model.uai");
        DataUtil.toCSV(folder + "/iv_data.csv", data);

        System.out.println("Model and data saved to: " + folder);
    }
}

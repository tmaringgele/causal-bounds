package repo.examples;

import java.io.*;
import java.util.*;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

public class CSVReader {

    public static TIntIntMap[] readCSV(File file) throws IOException {
        List<TIntIntMap> dataList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            // Skip header if present
            // line = br.readLine();
            while ((line = br.readLine()) != null) {
                String[] tokens = line.trim().split(",");
                TIntIntMap row = new TIntIntHashMap();
                for (int i = 0; i < tokens.length; i++) {
                    row.put(i, Integer.parseInt(tokens[i].trim()));
                }
                dataList.add(row);
            }
        }
        return dataList.toArray(new TIntIntMap[0]);
    }
}

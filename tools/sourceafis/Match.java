// SourceAFIS minutiae baseline: extract one template per image, then score listed pairs.
// Usage: java -cp "lib/*:." Match images.tsv pairs.tsv scores.txt
//   images.tsv: <index>\t<png path>\t<dpi>     pairs.tsv: <i>\t<j>     scores.txt: one score per pair
import com.machinezoo.sourceafis.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

public class Match {
    public static void main(String[] args) throws Exception {
        List<String[]> imgs = Files.readAllLines(Paths.get(args[0])).stream()
            .filter(s -> !s.isBlank()).map(s -> s.split("\t")).collect(Collectors.toList());
        FingerprintTemplate[] t = new FingerprintTemplate[imgs.size()];
        long t0 = System.currentTimeMillis();
        IntStream.range(0, imgs.size()).parallel().forEach(k -> {
            String[] r = imgs.get(k);
            try {
                var opt = new FingerprintImageOptions().dpi(Double.parseDouble(r[2]));
                t[Integer.parseInt(r[0])] = new FingerprintTemplate(new FingerprintImage(Files.readAllBytes(Paths.get(r[1])), opt));
            } catch (Exception e) { throw new RuntimeException(r[1], e); }
        });
        System.out.println("templates: " + t.length + " in " + (System.currentTimeMillis() - t0) / 1000.0 + " s");
        List<int[]> pairs = Files.readAllLines(Paths.get(args[1])).stream()
            .filter(s -> !s.isBlank()).map(s -> { String[] p = s.split("\t");
                return new int[]{Integer.parseInt(p[0]), Integer.parseInt(p[1])}; }).collect(Collectors.toList());
        // group by probe so each matcher is built once
        Map<Integer, List<Integer>> byProbe = IntStream.range(0, pairs.size()).boxed()
            .collect(Collectors.groupingBy(k -> pairs.get(k)[0]));
        double[] s = new double[pairs.size()];
        long t1 = System.currentTimeMillis();
        byProbe.entrySet().parallelStream().forEach(e -> {
            var m = new FingerprintMatcher(t[e.getKey()]);
            for (int k : e.getValue()) s[k] = m.match(t[pairs.get(k)[1]]);
        });
        System.out.println("pairs: " + pairs.size() + " in " + (System.currentTimeMillis() - t1) / 1000.0 + " s");
        StringBuilder sb = new StringBuilder();
        for (double v : s) sb.append(v).append('\n');
        Files.writeString(Paths.get(args[2]), sb.toString());
    }
}

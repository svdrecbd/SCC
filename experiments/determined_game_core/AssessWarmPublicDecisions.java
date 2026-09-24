import com.google.gson.*;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import owl.thirdparty.picocli.CommandLine;
import semml.semmlCommands.synthesisCommands.SemmlMain;

/** Reuse the JVM while constructing a fresh nonlearned solver for each formula. */
public final class AssessWarmPublicDecisions {
    public static void main(String[] arguments) throws Exception {
        Path root=Path.of(arguments[0]);
        JsonArray records=new Gson().fromJson(Files.readString(root.resolve("decisions.json")),JsonArray.class);
        JsonArray results=new JsonArray();
        org.tinylog.configuration.Configuration.set("level","ERROR");
        PrintStream original=System.out;
        for (var element : records) {
            JsonObject record=element.getAsJsonObject();
            List<String> command=new ArrayList<>();
            record.getAsJsonArray("command").forEach(value -> command.add(value.getAsString()));
            ByteArrayOutputStream captured=new ByteArrayOutputStream();
            int status;
            long started=System.nanoTime();
            try (PrintStream output=new PrintStream(captured,true,StandardCharsets.UTF_8)) {
                System.setOut(output);
                status=new CommandLine(new SemmlMain()).execute(command.subList(2,command.size()).toArray(String[]::new));
            } finally { System.setOut(original); }
            double seconds=(System.nanoTime()-started)/1e9;
            String text=captured.toString(StandardCharsets.UTF_8);
            var winners=text.lines().filter(line -> line.equals("REALIZABLE") || line.equals("UNREALIZABLE")).toList();
            JsonObject result=new JsonObject(); result.addProperty("name",record.get("name").getAsString());
            result.addProperty("seconds",seconds); result.addProperty("exit_code",status); result.addProperty("stdout",text);
            boolean matched=status==0 && winners.size()==1 && winners.get(0).equals(record.get("winner").getAsString());
            result.addProperty("matches_cold_decision",matched); results.add(result);
            Files.writeString(root.resolve("warm_decisions.json"),new GsonBuilder().setPrettyPrinting().create().toJson(results)+"\n");
            original.println(result);
            if (!matched) throw new AssertionError("Public solver disagrees with frozen decision.");
        }
    }
}

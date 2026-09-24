import com.google.gson.*;
import java.nio.file.*;
import java.util.*;
import owl.ltl.parser.LtlParser;
import owl.automaton.acceptance.GeneralizedBuchiAcceptance;
import owl.automaton.algorithm.LanguageEmptiness;
import owl.translations.ltl2nba.SymmetricNBAConstruction;

/** Check existence of an infinite input word satisfying each environment assumption. */
public final class VerifyTemporalAssumptions {
    public static void main(String[] arguments) throws Exception {
        JsonArray records = new Gson().fromJson(Files.readString(Path.of(arguments[0])),JsonArray.class);
        for (var element : records) {
            var record=element.getAsJsonObject();
            List<String> inputs=new ArrayList<>();
            record.getAsJsonArray("inputs").forEach(value -> inputs.add(value.getAsString()));
            List<String> clauses=new ArrayList<>();
            record.getAsJsonArray("assumptions").forEach(value -> clauses.add("("+value.getAsString()+")"));
            String formula=clauses.isEmpty() ? "true" : String.join(" & ",clauses);
            long started=System.nanoTime();
            var automaton=SymmetricNBAConstruction.of(GeneralizedBuchiAcceptance.class).apply(LtlParser.parse(formula,inputs));
            boolean satisfiable=!LanguageEmptiness.isEmpty(automaton);
            System.out.println(new Gson().toJson(Map.of("name",record.get("name").getAsString(),
                "assumptions_satisfiable",satisfiable,"seconds",(System.nanoTime()-started)/1e9)));
        }
    }
}

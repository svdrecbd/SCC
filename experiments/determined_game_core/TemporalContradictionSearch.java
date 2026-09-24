import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import owl.ltl.*;
import owl.ltl.parser.LtlParser;
import owl.ltl.visitors.Converter;

/** Sound incomplete contradiction certificates, without game solving or labels. */
public final class TemporalContradictionSearch {
    private static Formula assign(Formula formula, List<Integer> atoms, int assignment) {
        return formula.accept(new Converter(SyntacticFragment.ALL) {
            @Override public Formula visit(Literal literal) {
                int position = atoms.indexOf(literal.getAtom());
                if (position < 0) throw new IllegalArgumentException("Unassigned atom.");
                boolean value = (assignment & (1 << position)) != 0;
                return BooleanConstant.of(value != literal.isNegated());
            }
        });
    }
    private static Formula constantInputs(Formula formula, int count, BitSet values) {
        return formula.accept(new Converter(SyntacticFragment.ALL) {
            @Override public Formula visit(Literal literal) {
                return literal.getAtom() < count ? BooleanConstant.of(values.get(literal.getAtom()) != literal.isNegated()) : literal;
            }
        });
    }
    private static void conjuncts(Formula formula, List<Formula> result) {
        if (formula instanceof Conjunction) formula.operands.forEach(child -> conjuncts(child, result));
        else result.add(formula);
    }
    private record Requirement(Formula formula, int minimumTime, boolean recurring) {}
    private static Requirement requiredState(Formula formula) {
        int minimumTime = 0;
        boolean recurring = false;
        while (true) {
            if (formula instanceof XOperator next) { minimumTime++; formula=next.operand(); }
            else if (formula instanceof FOperator eventually) formula=eventually.operand();
            else if (formula instanceof GOperator always) { recurring=true; formula=always.operand(); }
            else break;
        }
        return SyntacticFragment.SINGLE_STEP.contains(formula) ? new Requirement(formula,minimumTime,recurring) : null;
    }
    private static List<BitSet> policies(int count) {
        List<BitSet> result = new ArrayList<>();
        BitSet zero = new BitSet();
        BitSet one = new BitSet(); one.set(0,count);
        result.add(zero); result.add(one);
        for (int index=0; index<count; index++) {
            BitSet sparseFalse=(BitSet)one.clone(); sparseFalse.clear(index); result.add(sparseFalse);
            BitSet sparseTrue=new BitSet(); sparseTrue.set(index); result.add(sparseTrue);
        }
        return result;
    }
    public static void main(String[] arguments) throws Exception {
        long deadline = System.nanoTime() + (long)(Double.parseDouble(arguments[1])*1e9);
        int maximumAtoms = Integer.parseInt(arguments[2]);
        var specifications = new Gson().fromJson(Files.readString(Path.of(arguments[0])), JsonArray.class);
        for (var element : specifications) {
            var specification = element.getAsJsonObject();
            String name = specification.get("name").getAsString();
            long started = System.nanoTime();
            Map<String,Object> result = new LinkedHashMap<>();
            result.put("name",name);
            result.put("result","UNKNOWN");
            if (System.nanoTime() >= deadline) {
                result.put("reason","stage_limit");
                System.out.println(new Gson().toJson(result));
                continue;
            }
            try {
            List<String> names = new ArrayList<>();
            specification.getAsJsonArray("inputs").forEach(value -> names.add(value.getAsString()));
            specification.getAsJsonArray("outputs").forEach(value -> names.add(value.getAsString()));
            for (BitSet constant : policies(specification.getAsJsonArray("inputs").size())) {
            if (System.nanoTime() >= deadline) break;
            boolean assumptionsHold = true;
            for (var property : specification.getAsJsonArray("assumptions")) {
                if (!constantInputs(LtlParser.parse(property.getAsString(),names).formula(),
                    specification.getAsJsonArray("inputs").size(),constant).equals(BooleanConstant.TRUE)) { assumptionsHold=false; break; }
            }
            if (!assumptionsHold) continue;
            List<Formula> clauses = new ArrayList<>();
            for (var property : specification.getAsJsonArray("guarantees"))
                conjuncts(constantInputs(LtlParser.parse(property.getAsString(),names).formula(),
                    specification.getAsJsonArray("inputs").size(), constant),clauses);
            List<Formula> invariants = new ArrayList<>();
            int invariantStart=0;
            for (Formula clause : clauses) {
                int start=0;
                Formula candidate=clause;
                while (candidate instanceof XOperator next) { start++; candidate=next.operand(); }
                if (candidate instanceof GOperator always && SyntacticFragment.SINGLE_STEP.contains(always.operand())) {
                    invariants.add(always.operand());
                    invariantStart=Math.max(invariantStart,start);
                }
            }
            Formula invariant = Conjunction.of(invariants);
            long checked = 0;
            for (Formula clause : clauses) {
                Requirement requirement = requiredState(clause);
                if (requirement == null || (!requirement.recurring() && requirement.minimumTime() < invariantStart)) continue;
                Formula obligation = requirement.formula();
                Formula joint = Conjunction.of(invariant, obligation);
                List<Integer> atoms = joint.atomicPropositions(true).stream().boxed().toList();
                if (atoms.size() > maximumAtoms) continue;
                boolean contradiction = true;
                boolean expired = false;
                for (int assignment=0; assignment<(1 << atoms.size()); assignment++) {
                    if (System.nanoTime() >= deadline) { expired=true; break; }
                    checked++;
                    Formula evaluated = assign(joint,atoms,assignment);
                    if (evaluated.equals(BooleanConstant.TRUE)) { contradiction=false; break; }
                    if (!evaluated.equals(BooleanConstant.FALSE)) throw new AssertionError("Residual non-Boolean formula.");
                }
                if (expired) break;
                if (contradiction) {
                    result.put("result","UNREALIZABLE");
                    result.put("constant_environment_true_indices",constant.stream().boxed().toList());
                    result.put("invariant_start",invariantStart);
                    result.put("requirement_recurring",requirement.recurring());
                    result.put("requirement_minimum_time",requirement.minimumTime());
                    result.put("invariants",invariants.stream().map(Object::toString).toList());
                    result.put("required_clause",clause.toString());
                    result.put("joint_state_requirement",joint.toString());
                    result.put("atoms",atoms);
                    result.put("valuation_count",1 << atoms.size());
                    break;
                }
            }
            result.put("valuations_checked",checked);
            if (result.get("result").equals("UNREALIZABLE")) break;
            }
            } catch (RuntimeException | StackOverflowError exception) {
                result.clear();
                result.put("name",name);
                result.put("result","UNKNOWN");
                result.put("reason","parse_or_resource_error");
                result.put("error",exception.toString());
            }
            result.put("seconds",(System.nanoTime()-started)/1e9);
            System.out.println(new Gson().toJson(result));
        }
    }
}

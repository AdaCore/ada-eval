with SPARK.Big_Integers; use SPARK.Big_Integers;
with Inst;
use Inst.Int_Sequences;

package Test
  with SPARK_Mode
is

   function Test_Add return Sequence
   with
     Post =>
       Last (Test_Add'Result) = To_Big_Integer (150)
       and then
         (for all K in 1 .. 150 =>
            Get (Test_Add'Result, To_Big_Integer (K)) = K);

end Test;

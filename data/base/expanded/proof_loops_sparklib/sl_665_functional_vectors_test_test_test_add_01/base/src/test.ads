with Inst;
use Inst.Int_Vectors;

package Test
  with SPARK_Mode
is

   function Test_Add return Sequence
   with
     Post =>
       Last (Test_Add'Result) = 150
       and then (for all K in 1 .. 150 => Get (Test_Add'Result, K) = K);

end Test;

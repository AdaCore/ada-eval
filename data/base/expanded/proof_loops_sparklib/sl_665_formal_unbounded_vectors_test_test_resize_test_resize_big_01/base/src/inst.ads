with SPARK.Containers.Formal.Unbounded_Vectors;

package Inst with SPARK_Mode is
   function Eq (X, Y : Integer) return Boolean is (X mod 1000 = Y mod 1000);
   package Int_Vectors is new SPARK.Containers.Formal.Unbounded_Vectors
     (Positive, Integer, "=" => Eq);
end Inst;

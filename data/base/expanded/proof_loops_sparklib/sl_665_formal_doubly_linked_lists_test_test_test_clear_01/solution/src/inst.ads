with SPARK.Containers.Formal.Doubly_Linked_Lists;

package Inst
  with SPARK_Mode
is
   function Eq (X, Y : Integer) return Boolean
   is (X mod 1000 = Y mod 1000);
   package Int_Lists is new
     SPARK.Containers.Formal.Doubly_Linked_Lists (Integer, "=" => Eq);
end Inst;

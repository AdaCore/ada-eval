--  Instantiation of the unit under analysis.  GNATprove analyses instances,
--  not generic units, so the extracted loop needs one.  The actuals are
--  upstream's own, from tests/W127-011__functional_sets_HO_proof.

pragma Ada_2022;
with SPARK.Big_Integers; use SPARK.Big_Integers;
with SPARK.Containers.Functional.Sets;
with SPARK.Containers.Functional.Sets.Higher_Order;

package Inst with SPARK_Mode is

   package Nested with
      Always_Terminates
   is
      type Element_Type is private;

      function Witness (E : Element_Type) return Big_Integer;
      function Witness (W : Big_Integer) return Big_Integer;

      function "=" (Left, Right : Element_Type) return Boolean with
        Post => "="'Result = (Witness (Left) = Witness (Right));
      function Equivalent_Elements
        (Left  : Element_Type;
         Right : Element_Type) return Boolean
      with
        Post => Equivalent_Elements'Result =
            (Witness (Witness (Left)) = Witness (Witness (Right)));

   private
      pragma SPARK_Mode (Off);

      type Element_Type is new Big_Integer;

      function Witness (E : Element_Type) return Big_Integer is
        (Big_Integer (E));
      function Witness (W : Big_Integer) return Big_Integer is
        (W);

      function "=" (Left, Right : Element_Type) return Boolean is
        (Witness (Left) = Witness (Right));
      function Equivalent_Elements
        (Left  : Element_Type;
         Right : Element_Type) return Boolean
      is (Left = Right);
   end Nested;
   use Nested;

   package My_Sets is new SPARK.Containers.Functional.Sets
     (Element_Type, Equivalent_Elements);

   package HO is new My_Sets.Higher_Order;

end Inst;

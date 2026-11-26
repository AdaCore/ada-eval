with Ada.Numerics.Big_Numbers.Big_Integers;
use Ada.Numerics.Big_Numbers.Big_Integers;
with SPARK.Higher_Order;
with SPARK.Higher_Order.Fold;
with Test_Types;                            use Test_Types;

package Test_Higher_Order2
  with SPARK_Mode
is

   function Value (X : Integer) return My_Small
   with Import, Global => null;

   --  Value cannot have a pre as it is called from Add_Value without any
   --  constraint.

   function In_Range (X : Big_Integer) return Boolean
   with Import, Global => null, Ghost;

   function To_Big (X : Integer) return Big_Integer
   with Import, Global => null, Ghost;

   function Add_Pre (X, Y : Integer) return Boolean
   with Import, Global => null;

   function Add (X, Y : Integer) return Integer
   with Import, Global => null, Pre => Add_Pre (X, Y);

   package My_Sum is new
     SPARK.Higher_Order.Fold.Sum
       (Index_Type  => Small_Index,
        Element_In  => Integer,
        Array_Type  => Small_Array,
        Element_Out => Integer,
        Zero        => 0,
        Add         => Add,
        To_Big      => To_Big,
        In_Range    => In_Range,
        Value       => Value);

end Test_Higher_Order2;

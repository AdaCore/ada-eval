--  Element type for the vector representation.  It is a private type whose
--  full view is hidden from SPARK analysis and "<" is an imported function
--  with no contract, so the loop's proof cannot rely on any property of the
--  element type or of the ordering.  Copied from what the upstream proof
--  test (tests/1376__formal_vectors_impl_proof) supplies.

package Element_Type_Defs with SPARK_Mode is

   type Element_Type is private;

   function "<" (Left, Right : Element_Type) return Boolean
   with Import, Global => null;

private
   pragma SPARK_Mode (Off);

   type Element_Type is new Integer;

end Element_Type_Defs;

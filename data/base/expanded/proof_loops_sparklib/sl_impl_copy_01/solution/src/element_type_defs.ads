--  Element type for instantiating SPARK.Containers.Formal.Doubly_Linked_Lists
--  .Impl in the proof test. It is a private type whose full view is hidden
--  from SPARK analysis (the private part is SPARK_Mode => Off), and "=" and
--  "<" are imported functions with no contracts. The proof then cannot rely on
--  any property of the elements beyond the contracts.

package Element_Type_Defs with SPARK_Mode is

   type Element_Type is private;

   function "=" (Left, Right : Element_Type) return Boolean
   with Import, Global => null;

private
   pragma SPARK_Mode (Off);

   type Element_Type is new Integer;

end Element_Type_Defs;

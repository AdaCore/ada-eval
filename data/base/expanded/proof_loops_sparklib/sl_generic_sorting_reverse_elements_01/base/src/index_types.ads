--  Index type for the vector representation.  Copied from what the
--  upstream proof test (tests/1376__formal_vectors_impl_proof) supplies for
--  the "wide index" shape: the base type is Long_Long_Integer (the widest base
--  type the implementation supports), and the subtype bounds are obtained
--  through function calls with no precise contract, so they are not static.

package Index_Types
  with SPARK_Mode
is

   --  Forces Long_Long_Integer as the base type (the static range exceeds
   --  32 bits but stays within 64 bits).
   type Big_Base is range -(2 ** 62) .. 2 ** 62;

   --  First is not the first element of its base type (to allow the definition
   --  of Extended_Index), and there is at least a valid index.

   function First return Big_Base
   with Import, Global => null, Post => First'Result /= Big_Base'Base'First;

   function Last return Big_Base
   with Import, Global => null, Post => Last'Result >= First;

   Big_First : constant Big_Base := First;
   Big_Last  : constant Big_Base := Last;

   subtype Big_Index is Big_Base range Big_First .. Big_Last;

end Index_Types;

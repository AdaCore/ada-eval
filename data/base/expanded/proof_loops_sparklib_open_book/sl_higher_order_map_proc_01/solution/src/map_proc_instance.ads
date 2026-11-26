with SPARK.Higher_Order;

package Map_Proc_Instance
  with SPARK_Mode
is

   --  SPARK.Higher_Order.Map_Proc is a generic subprogram, and GNATprove
   --  analyses only instances.  Upstream never instantiates it, so these are
   --  the simplest actuals that typecheck: an integer index type, Integer
   --  elements, an opaque F and an opaque ghost Init_Prop.

   type Index_Type is range 0 .. 1_000;

   type Array_Type is array (Index_Type range <>) of Integer;

   function Init_Prop (A : Integer) return Boolean
   with Ghost, Import, Global => null;

   function F (X : Integer) return Integer
   with Import, Global => null;

   procedure Map_Integers is new
     SPARK.Higher_Order.Map_Proc
       (Index_Type => Index_Type,
        Element    => Integer,
        Array_Type => Array_Type,
        Init_Prop  => Init_Prop,
        F          => F);

end Map_Proc_Instance;

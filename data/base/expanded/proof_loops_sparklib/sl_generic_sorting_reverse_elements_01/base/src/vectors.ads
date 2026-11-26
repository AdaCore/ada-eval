--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

--  A local model of the bounded-vector representation that
--  SPARK.Containers.Formal.Vectors declares in its private part and that
--  SPARK.Containers.Formal.Vectors.Impl operates on.  Only what
--  Reverse_Elements touches is kept; the representation itself
--  (Relaxed_Element,
--  Elements_Array, To_Array_Index, the Vector record and its Ghost_Predicate)
--  is copied verbatim from upstream.

pragma Ada_2022;

with SPARK.Containers.Types; use SPARK.Containers.Types;
with Element_Type_Defs;      use Element_Type_Defs;
with Index_Types;

package Vectors
  with SPARK_Mode, Always_Terminates
is

   subtype Index_Type is Index_Types.Big_Index;
   use type Index_Types.Big_Base;
   --  Upstream Index_Type is a generic formal signed type, whose operators are
   --  directly visible; here it is a named subtype, so the operators of its
   --  base type have to be made visible explicitly.

   subtype Extended_Index is
     Index_Type'Base
       range Index_Type'Pred (Index_Type'First) .. Index_Type'Last;

   No_Index : constant Extended_Index := Extended_Index'First;

   Last_Count : constant Count_Type :=
     (if Index_Type'Last < Index_Type'First
      then 0
      elsif (if Index_Type'First > 0
             then
               Index_Type'Pos (Index_Type'Last)
               - Index_Type'Pos (Index_Type'First)
               < Count_Type'Pos (Count_Type'Last)
             else
               Index_Type'Pos (Index_Type'Last)
               < Count_Type'Pos (Count_Type'Last)
                 + Index_Type'Pos (Index_Type'First))
      then
        Count_Type'Val
          (Index_Type'Pos (Index_Type'Last)
           - Index_Type'Pos (Index_Type'First)
           + 1)
      else Count_Type'Last);
   --  Maximal capacity of any vector. It is the minimum of the size of the
   --  index range and the last possible Count_Type.

   subtype Capacity_Range is Count_Type range 0 .. Last_Count;

   type Vector (Capacity : Capacity_Range) is private;

   function Length (Container : Vector) return Capacity_Range
   with Global => null;

   procedure Reverse_Elements (Container : in out Vector)
   with Global => null, Post => Length (Container) = Length (Container)'Old;

private

   --  The payload is wrapped in a one-field record marked
   --  Relaxed_Initialization, so that only the used prefix Elements
   --  (1 .. Last) needs to be initialized; the unused cells hold uninitialized
   --  Element_Type values. The V field is aliased so that Reference and
   --  Constant_Reference can hand out an access designating it.

   type Relaxed_Element is record
      V : aliased Element_Type;
   end record
   with Relaxed_Initialization;

   subtype Array_Index is Capacity_Range range 1 .. Capacity_Range'Last;
   type Elements_Array is array (Array_Index range <>) of Relaxed_Element;

   function "=" (L, R : Elements_Array) return Boolean is abstract;

   function To_Array_Index (Index : Index_Type'Base) return Count_Type'Base
   is
      --  We know that
      --    Index >= Index_Type'First
      --  hence we also know that
      --    Index - Index_Type'First >= 0

      --  The issue is that even though 0 is guaranteed to be a value in
      --  the type Index_Type'Base, there's no guarantee that the difference
      --  is a value in that type. To prevent overflow we use the wider
      --  of Count_Type'Base and Index_Type'Base to perform intermediate
      --  calculations.

      --  The array index subtype for all container element arrays always
      --  starts with 1.

      (if Index_Type'Base'Last >= Count_Type'Pos (Count_Type'Last)
       then Count_Type'Base (Index - No_Index)
       else Count_Type'Base (Index) - Count_Type'Base (No_Index))
   with
     Pre =>
       Index >= No_Index
       and then Index <= No_Index + Count_Type'Pos (Last_Count);

   type Vector (Capacity : Capacity_Range) is record
      Last     : Extended_Index := No_Index;
      Elements : Elements_Array (1 .. Capacity);
   end record
   with
     Ghost_Predicate =>
       (Static =>
          Last <= No_Index + Count_Type'Pos (Last_Count)
          and then To_Array_Index (Last) <= Capacity
          and then
            (for all I in 1 .. To_Array_Index (Last) =>
               Elements (I).V'Initialized));

   function Length (Container : Vector) return Capacity_Range
   is (To_Array_Index (Container.Last));

end Vectors;

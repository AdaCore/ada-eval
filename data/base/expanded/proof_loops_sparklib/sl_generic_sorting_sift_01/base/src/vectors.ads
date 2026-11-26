--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

--  A local model of the bounded-vector representation that Generic_Sorting
--  works on, copied from SPARK.Containers.Formal.Vectors (private part) and
--  from SPARK.Containers.Formal.Vectors.Impl (Length).
--
--  Everything else -- Extended_Index, No_Index, Last_Count, Capacity_Range,
--  Relaxed_Element, Array_Index, Elements_Array, To_Array_Index, the Vector
--  record with its Ghost_Predicate, and Length -- is copied verbatim.

pragma Ada_2022;

with SPARK.Containers.Types; use SPARK.Containers.Types;

package Vectors with SPARK_Mode is

   subtype Index_Type is Positive;
   --  Generic formal of SPARK.Containers.Formal.Vectors.

   type Element_Type is private;
   --  Generic formal of SPARK.Containers.Formal.Vectors.

   function "<" (Left, Right : Element_Type) return Boolean
   with Import, Global => null;
   --  Generic formal of Generic_Sorting; opaque here.

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
   --  The used prefix fits in the capacity and is fully initialized.

   function Length (Container : Vector) return Capacity_Range
   is (To_Array_Index (Container.Last))
   with Global => null;

private

   type Element_Type is record
      X : Integer;
   end record;

end Vectors;

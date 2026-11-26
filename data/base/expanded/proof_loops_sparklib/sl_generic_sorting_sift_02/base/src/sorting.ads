--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

--  Generic_Sorting.Sort's sift-up loop, isolated.
--
--  Upstream, Sift is nested inside Generic_Sorting.Sort and reaches Container,
--  Max and Temp as up-level references to Sort's parameter and locals.  Here
--  they are parameters of Sift, which leaves the loop text unchanged.  Is_Anc
--  and T are Sort's own declarations, copied verbatim.

pragma Ada_2022;

with SPARK.Containers.Types; use SPARK.Containers.Types;
with Vectors; use Vectors;

package Sorting with SPARK_Mode, Always_Terminates is

   subtype T is Long_Long_Integer;

   function Is_Anc (Root, Node : T) return Boolean
   is (Node = Root
       or else (Node > Root and then Is_Anc (Root, Node / 2)))
   with
     Ghost              => Static,
     Global             => null,
     Subprogram_Variant => (Decreases => Node),
     Pre                => (Static => Root >= 1 and then Node >= 1);
   --  Node lies in the subtree rooted at Root, i.e. Root is reached by
   --  halving Node repeatedly.

   procedure Sift
     (Container : in out Vector;
      S         : T;
      Max       : T;
      Temp      : Element_Type;
      C0        : T)
   with
     Global => null,
     Pre  =>
       (Static =>
          S in 1 .. Max
          and then Max <= T (Length (Container))
          and then C0 in S .. Max
          and then Is_Anc (S, C0)),
     Post => (Static => Length (Container) = Length (Container)'Old);
   --  Sift the element held in Temp up the heap path from C0 to the root S.
   --  Only permutes Elements (1 .. Max); leaves the vector's length unchanged.

end Sorting;

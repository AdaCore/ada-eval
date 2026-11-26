--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
procedure Free (Container : in out List; X : Count_Type) is

   N     : Node_Array renames Container.Nodes;
   M_Old : constant Nodes_Type_Base := Memory (Container)
   with Ghost => Static;

begin
   N (X).Prev := -1;  -- Node is deallocated (not on active list)

   if Container.Free >= 0 then
      N (X).Next := Container.Free;

      Lemma_Reachable_Preserved
        (Container.First, M_Old, Memory (Container));
      Lemma_Is_Acyclic_Preserved
        (Container.First, M_Old, Memory (Container));

      --  Writing N (X) leaves the old free chain untouched, so it is
      --  preserved; push X onto its head to obtain the new free chain.

      Lemma_Reachable_Preserved (Container.Free, M_Old, Memory (Container));
      Lemma_Is_Acyclic_Preserved
        (Container.Free, M_Old, Memory (Container));
      Lemma_Is_Acyclic_Def (X, Memory (Container));
      Lemma_Reachable_Def (X, Memory (Container));

      Container.Free := X;

   elsif Container.Free = -X - 1 then
      Lemma_Reachable_Closed_By_Next (Container.First, M_Old);

      N (X).Next := 0;

      Container.Free := Container.Free + 1;

      Lemma_Reachable_Preserved
        (Container.First, M_Old, Memory (Container));
      Lemma_Is_Acyclic_Preserved
        (Container.First, M_Old, Memory (Container));

      --  Free stays negative, so the free chain is still the implicit tail
      --  and Free_Chain_Valid holds trivially.

   elsif Container.Free = -Container.Capacity - 1 then

      --  The never-used region is empty (every cell has been handed out via
      --  the negative-Free encoding), so the rebuilt free list is just X.

      N (X).Next := 0;
      Container.Free := X;

      Lemma_Reachable_Preserved
        (Container.First, M_Old, Memory (Container));
      Lemma_Is_Acyclic_Preserved
        (Container.First, M_Old, Memory (Container));

      Lemma_Is_Acyclic_Def (X, Memory (Container));
      Lemma_Reachable_Def (X, Memory (Container));

   else
      declare
         Base : constant Count_Type := -Container.Free;
      begin
         for J in Base .. Container.Capacity - 1 loop
            N (J).Next := J + 1;
            N (J).Prev := -1;
         end loop;

         N (Container.Capacity).Next := 0;
         N (Container.Capacity).Prev := -1;

         Container.Free := X;
         N (X).Next := Base;

         Lemma_Reachable_Preserved
           (Container.First, M_Old, Memory (Container));
         Lemma_Is_Acyclic_Preserved
           (Container.First, M_Old, Memory (Container));

         --  Reconstruct the rebuilt free list, now headed at X (X -> Base
         --  -> ... -> Capacity -> 0), via the local lemma.

         Rebuild_Free_List (Container, X, Base);
      end;
   end if;
end Free;

--
--  Copyright (C) 2004-2026, Free Software Foundation, Inc.
--
--  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
--

pragma Ada_2022;
separate (Lists)
function Reverse_Find
  (Container : List; Item : Element_Type; Position : Cursor := No_Element)
   return Cursor
is
   CFirst : Count_Type := Position.Node;

begin
   if CFirst = 0 then
      CFirst := Container.Last;
   end if;

   if Container.Length = 0 then
      return No_Element;

   else
      while CFirst /= 0 loop
         Lemma_Reachable_Is_Acyclic
           (Container.First, CFirst, Memory (Container));
         Lemma_Reachable_Def (CFirst, Memory (Container));
         Lemma_Reachable_Included
           (Container.First, CFirst, Memory (Container));
         pragma
           Assert
             (Static =>
                Memory_Index_Sets.Num_Overlaps
                  (Reachable_Set (CFirst, Memory (Container)),
                   Reachable_Set (Container.First, Memory (Container)))
                = Memory_Index_Sets.Length
                    (Reachable_Set (CFirst, Memory (Container))));

         pragma Warnings (Off, "condition is always True");
         pragma Loop_Invariant (Static => CFirst /= 0);
         pragma
           Loop_Invariant
             (Static =>
                Reachable (Container.First, Memory (Container), CFirst));
         pragma
           Loop_Variant
             (Static =>
                (Decreases =>
                   Memory_Index_Sets.Length
                     (Reachable_Set (Container.First, Memory (Container)))
                   - Memory_Index_Sets.Length
                       (Reachable_Set (CFirst, Memory (Container)))));
         pragma Warnings (On, "condition is always True");

         if Container.Nodes (CFirst).Element = Item then
            return (Node => CFirst);
         else
            CFirst := Container.Nodes (CFirst).Prev;
         end if;
      end loop;

      return No_Element;
   end if;
end Reverse_Find;

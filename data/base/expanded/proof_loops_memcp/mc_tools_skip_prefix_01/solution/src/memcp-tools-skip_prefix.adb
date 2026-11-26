separate (Memcp.Tools)
function Skip_Prefix (S : String; P : Positive) return Natural is
   R : Positive := P;
begin
   if S (P) not in 'A' .. 'Z' | 'a' .. 'z' then
      return P;   --  not a prefix start: the tag name begins at P

   end if;
   --  Advance over the prefix token [A-Za-z][A-Za-z0-9_.-]*.
   while R < S'Last
     and then
       S (R + 1) in 'A' .. 'Z' | 'a' .. 'z' | '0' .. '9' | '_' | '.' | '-'
   loop
      pragma Loop_Invariant (R in S'Range);
      pragma Loop_Variant (Increases => R);
      R := R + 1;
   end loop;
   --  A ':' immediately after makes P .. R a namespace prefix.
   if R < S'Last and then S (R + 1) = ':' then
      return After (S, R + 1);
   end if;
   return P;   --  no terminating colon: not a prefix, name starts at P
end Skip_Prefix;

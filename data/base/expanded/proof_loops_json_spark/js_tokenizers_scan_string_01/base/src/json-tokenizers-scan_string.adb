--  SPDX-License-Identifier: Apache-2.0
--
--  Copyright (c) 2016 onox <denkpadje@gmail.com>
--
--  Licensed under the Apache License, Version 2.0 (the "License");
--  you may not use this file except in compliance with the License.
--  You may obtain a copy of the License at
--
--      http://www.apache.org/licenses/LICENSE-2.0
--
--  Unless required by applicable law or agreed to in writing, software
--  distributed under the License is distributed on an "AS IS" BASIS,
--  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
--  See the License for the specific language governing permissions and
--  limitations under the License.

separate (JSON.Tokenizers)
procedure Scan_String
  (Stream     : in out Streams.Stream;
   Next_Token : out Token;
   Status     : out Token_Status)
is
   C       : Character;
   EOF     : Boolean;
   Index   : Positive;
   Count   : Natural := 0;
   Escaped : Boolean := False;
begin
   Next_Token := (Kind => Invalid_Token, others => <>);
   Status := Success;

   loop
      pragma Loop_Variant (Increases => Streams.Position (Stream));

      Streams.Read_Character (Stream, Index, C, EOF);
      if EOF then
         Status := Error_Unexpected_EOF;
         return;
      end if;

      --  An unescaped '"' character denotes the end of the string
      exit when not Escaped and C = '"';

      Count := Count + 1;

      if Escaped then
         case C is
            when '"' | '\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' =>
               null;

            when 'u'                                           =>
               --  A \uXXXX escape sequence: scan the 4 hexadecimal
               --  digits. JSON strings encode code points beyond the
               --  Basic Multilingual Plane as a UTF-16 surrogate
               --  pair (RFC 8259 section 7), so a high surrogate
               --  must be followed by \u and a low surrogate, and a
               --  lone surrogate is invalid.
               declare
                  Unit : Natural;
               begin
                  Scan_Hex_Quad (Stream, Count, Unit, Status);
                  if Status /= Success then
                     return;
                  end if;

                  if Unit in 16#DC00# .. 16#DFFF# then
                     --  A low surrogate without a preceding high
                     --  surrogate
                     Status := Error_Escaped_Unicode;
                     return;
                  elsif Unit in 16#D800# .. 16#DBFF# then
                     Streams.Read_Character (Stream, C, EOF);
                     if EOF then
                        Status := Error_Unexpected_EOF;
                        return;
                     end if;
                     Count := Count + 1;
                     if C /= '\' then
                        Status := Error_Escaped_Unicode;
                        return;
                     end if;

                     Streams.Read_Character (Stream, C, EOF);
                     if EOF then
                        Status := Error_Unexpected_EOF;
                        return;
                     end if;
                     Count := Count + 1;
                     if C /= 'u' then
                        Status := Error_Escaped_Unicode;
                        return;
                     end if;

                     Scan_Hex_Quad (Stream, Count, Unit, Status);
                     if Status /= Success then
                        return;
                     end if;
                     if Unit not in 16#DC00# .. 16#DFFF# then
                        Status := Error_Escaped_Unicode;
                        return;
                     end if;
                  end if;
               end;

            when others                                        =>
               Status := Error_Escaped_Character;
               return;
         end case;
      elsif C /= '\' then
         --  Check C is not a control character
         if Character'Pos (C) <= 31 then
            Status := Error_Control_Character;
            return;
         end if;
      end if;
      Escaped := not Escaped and C = '\';
   end loop;

   Next_Token :=
     (Kind          => String_Token,
      String_Offset => Index - Count,
      String_Length => Count,
      others        => <>);
end Scan_String;

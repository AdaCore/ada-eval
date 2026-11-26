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

separate (JSON.Parsers)
procedure Parse_Object
  (Stream        : in out Streams.Stream;
   Depth         : Positive;
   Maximum_Depth : Positive;
   Result        : out Types.JSON_Value_Access;
   Status        : out Parse_Status;
   Detail        : out Tokenizers.Token_Status)
is
   Container : Types.JSON_Value_Access := Types.Create_Object;
   Element   : Types.JSON_Value_Access;

   Token        : Tokenizers.Token;
   Token_Status : Tokenizers.Token_Status;

   Repeat : Boolean := False;
begin
   Result := null;
   Status := Ok;
   Detail := Tokenizers.Success;

   loop
      pragma
        Loop_Variant
          (Decreases => Streams.Length (Stream) - Streams.Position (Stream),
           Decreases =>
             (if Streams.Has_Buffered_Character (Stream) then 1 else 0));

      Tokenizers.Scan_Token (Stream, Token, Token_Status);
      if Token_Status /= Tokenizers.Success then
         Types.Free (Container);
         Status := Tokenizer_Failed;
         Detail := Token_Status;
         return;
      end if;

      --  Either expect '}' character or (if not the first member)
      --  a value separator (',' character)
      if Token.Kind = Tokenizers.End_Object_Token then
         exit;
      elsif Repeat then
         if Token.Kind /= Tokenizers.Value_Separator_Token then
            Types.Free (Container);
            Status := Expected_Value_Separator;
            return;
         end if;

         --  Value separator has been read, now read the next member
         Tokenizers.Scan_Token (Stream, Token, Token_Status);
         if Token_Status /= Tokenizers.Success then
            Types.Free (Container);
            Status := Tokenizer_Failed;
            Detail := Token_Status;
            return;
         end if;
      end if;

      --  Parse the member key
      if Token.Kind /= Tokenizers.String_Token then
         Types.Free (Container);
         Status := Expected_String_Key;
         return;
      end if;

      declare
         Member_Key : constant String :=
           Streams.Get_String
             (Stream, Token.String_Offset, Token.String_Length);
      begin
         if Check_Duplicate_Keys
           and then Types.Contains (Container, Member_Key)
         then
            Types.Free (Container);
            Status := Duplicate_Key;
            return;
         end if;

         --  Expect a name separator (':' character) between the key
         --  and the value
         Tokenizers.Scan_Token (Stream, Token, Token_Status);
         if Token_Status /= Tokenizers.Success then
            Types.Free (Container);
            Status := Tokenizer_Failed;
            Detail := Token_Status;
            return;
         end if;

         if Token.Kind /= Tokenizers.Name_Separator_Token then
            Types.Free (Container);
            Status := Expected_Name_Separator;
            return;
         end if;

         --  Parse the member value and prepend the member
         Tokenizers.Scan_Token (Stream, Token, Token_Status);
         if Token_Status /= Tokenizers.Success then
            Types.Free (Container);
            Status := Tokenizer_Failed;
            Detail := Token_Status;
            return;
         end if;

         Parse_Value
           (Stream, Token, Depth, Maximum_Depth, Element, Status, Detail);
         if Status /= Ok then
            Types.Free (Container);
            return;
         end if;

         if Types.Length (Container) = Natural'Last then
            Types.Free (Container);
            Types.Free (Element);
            Status := Too_Many_Elements;
            return;
         end if;
         Types.Prepend_Member (Container, Member_Key, Element);
      end;

      Repeat := True;
   end loop;

   Types.Reverse_Elements (Container);
   Result := Container;
end Parse_Object;

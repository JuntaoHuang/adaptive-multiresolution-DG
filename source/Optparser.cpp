#include "Optparser.h"

int isValidAsInt(char * s)
{
   if ( s == NULL || *s == '\0' )
   {
      return 0;   //Empty string
   }

   if ( *s == '+' || *s == '-' )
   {
      ++s;
   }

   if ( *s == '\0')
   {
      return 0;   //sign character only
   }

   while (*s)
   {
      if ( !isdigit(*s) )
      {
         return 0;
      }
      ++s;
   }

   return 1;
}

int isValidAsDouble(char * s)
{
   //A valid floating point number for atof using the "C" locale is formed by
   // - an optional sign character (+ or -),
   // - followed by a sequence of digits, optionally containing a decimal-point
   //   character (.),
   // - optionally followed by an exponent part (an e or E character followed by
   //   an optional sign and a sequence of digits).

   if ( s == NULL || *s == '\0' )
   {
      return 0;   //Empty string
   }

   if ( *s == '+' || *s == '-' )
   {
      ++s;
   }

   if ( *s == '\0')
   {
      return 0;   //sign character only
   }

   while (*s)
   {
      if (!isdigit(*s))
      {
         break;
      }
      ++s;
   }

   if (*s == '\0')
   {
      return 1;   //s = "123"
   }

   if (*s == '.')
   {
      ++s;
      while (*s)
      {
         if (!isdigit(*s))
         {
            break;
         }
         ++s;
      }
      if (*s == '\0')
      {
         return 1;   //this is a fixed point double s = "123." or "123.45"
      }
   }

   if (*s == 'e' || *s == 'E')
   {
      ++s;
      return isValidAsInt(s);
   }
   else
   {
      return 0;   //we have encounter a wrong character
   }
}

void OptionsParser::Parse()
{
   option_check.resize(options.size());
   for (auto & i : option_check) { i = 0; }
   
   for (int i = 1; i < argc; )
   {
      if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
      {
         // print help message
         error_type = 1;
         return;
      }

      for (int j = 0; true; j++)
      {
         if (j >= options.size())
         {
            // unrecognized option
            error_type = 2;
            error_idx = i;
            return;
         }

         if (strcmp(argv[i], options[j].short_name) == 0 ||
             strcmp(argv[i], options[j].long_name) == 0)
         {
            OptionType type = options[j].type;

            if ( option_check[j] )
            {
               error_type = 4;
               error_idx = j;
               return;
            }
            option_check[j] = 1;

            i++;

            int isValid = 1;
            switch (options[j].type)
            {
               case INT:
                  isValid = isValidAsInt(argv[i]);
                  *(int *)(options[j].var_ptr) = atoi(argv[i++]);
                  break;
               case DOUBLE:
                  isValid = isValidAsDouble(argv[i]);
                  *(double *)(options[j].var_ptr) = atof(argv[i++]);
                  break;
               case STRING:
                  *(const char **)(options[j].var_ptr) = argv[i++];
                  break;
            }

            if (!isValid)
            {
               error_type = 5;
               error_idx = i;
               return;
            }

            break;
         }
      }
   }

   // check for missing required options
   for (int i = 0; i < options.size(); i++)
      if (options[i].required && option_check[i] == 0)
      {
         error_type = 6; // required option missing
         error_idx = i; // for a boolean option i is the index of DISABLE
         return;
      }

   error_type = 0;
}

void OptionsParser::WriteValue(const Option &opt, std::ostream &out)
{
   switch (opt.type)
   {
      case INT:
         out << *(int *)(opt.var_ptr);
         break;

      case DOUBLE:
         out << *(double *)(opt.var_ptr);
         break;

      case STRING:
         out << *(const char **)(opt.var_ptr);
         break;

      default: // provide a default to suppress warning
         break;
   }
}

void OptionsParser::PrintOptions(std::ostream &out) const
{
   static const char *indent = "   ";

   out << "Options used:\n";
   for (int j = 0; j < options.size(); j++)
   {
      OptionType type = options[j].type;

      out << indent;
      out << std::setw(5) << options[j].short_name << " ";
      out << "    (" << options[j].long_name << ") "; 
      WriteValue(options[j], out);
      out << '\n';
   }
}

void OptionsParser::PrintError(std::ostream &out) const
{
   static const char *line_sep = "";

   out << line_sep;
   switch (error_type)
   {
      case 2:
         out << "Unrecognized option: " << argv[error_idx] << '\n' << line_sep;
         break;

      case 3:
         out << "Missing argument for the last option: " << argv[argc-1] << '\n'
             << line_sep;
         break;

      case 4:
         out << "Option " << options[error_idx].long_name
             << " provided multiple times\n" << line_sep;
         break;

      case 5:
         out << "Wrong option format: " << argv[error_idx - 1] << " "
             << argv[error_idx] << '\n' << line_sep;
         break;

      case 6:
         out << "Missing required option: " << options[error_idx].long_name
             << '\n' << line_sep;
         break;
   }
   out << std::endl;
}

void OptionsParser::PrintHelp(std::ostream &out) const
{
   static const char *indent = "   ";
   static const char *seprtr = ", ";
   static const char *descr_sep = "\n\t";
   static const char *line_sep = "";
   static const char *types[] = { " <int>", " <double>", " <string>", "", "",
                                  " '<int>...'", " '<double>...'"
                                };

   out << indent << "-h" << seprtr << "--help" << descr_sep
       << "Print this help message and exit.\n" << line_sep;
   for (int j = 0; j < options.size(); j++)
   {
      OptionType type = options[j].type;

      out << indent << options[j].short_name << types[type]
          << seprtr << options[j].long_name << types[type]
          << seprtr;
      if (options[j].required)
      {
         out << "(required)";
      }
      else
      {
         out << "current value: ";
         WriteValue(options[j], out);
      }
      out << descr_sep;

      if (options[j].description)
      {
         out << options[j].description << '\n';
      }
      out << line_sep;
   }
}

void OptionsParser::PrintUsage(std::ostream &out) const
{
   static const char *line_sep = "";

   PrintError(out);
   out << "Usage: " << argv[0] << " [options] ...\n" << line_sep
       << "Options:\n" << line_sep;
   PrintHelp(out);
}
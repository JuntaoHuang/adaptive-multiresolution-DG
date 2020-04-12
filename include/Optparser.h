#pragma once
#include "libs.h"

/** 
   Class for parsing command-line options adapted from MFEM (http://mfem.org)
   The class is initialized with argc and argv, and new options are added with
   the AddOption method. Currently options of type int, double, char* supported.
*/
class OptionsParser
{
public:
   enum OptionType { INT, DOUBLE, STRING};

private:
   struct Option
   {
      OptionType type;
      void *var_ptr;
      const char *short_name;
      const char *long_name;
      const char *description;
      bool required;

      Option() = default;

      Option(OptionType _type, void *_var_ptr, const char *_short_name,
             const char *_long_name, const char *_description, bool req)
         : type(_type), var_ptr(_var_ptr), short_name(_short_name),
           long_name(_long_name), description(_description), required(req) { }
   };

   int argc;
   char **argv;
   std::vector<Option> options;
   std::vector<int> option_check;
   // error_type can be:
   //  0 - no error
   //  1 - print help message
   //  2 - unrecognized option at argv[error_idx]
   //  3 - missing argument for the last option argv[argc-1]
   //  4 - option with index error_idx is specified multiple times
   //  5 - invalid argument in argv[error_idx] for option in argv[error_idx-1]
   //  6 - required option with index error_idx is missing
   int error_type, error_idx;

   static void WriteValue(const Option &opt, std::ostream &out);

public:
   OptionsParser(int _argc, char *_argv[])
      : argc(_argc), argv(_argv)
   {
      error_type = error_idx = 0;
   }
   
   void AddOption(int *var, const char *short_name, const char *long_name,
                  const char *description, bool required = false)
   {
      options.push_back(Option(INT, var, short_name, long_name, description,
                            required));
   }
   void AddOption(double *var, const char *short_name, const char *long_name,
                  const char *description, bool required = false)
   {
      options.push_back(Option(DOUBLE, var, short_name, long_name, description,
                            required));
   }
   void AddOption(const char **var, const char *short_name,
                  const char *long_name, const char *description,
                  bool required = false)
   {
      options.push_back(Option(STRING, var, short_name, long_name, description,
                            required));
   }

   void Parse();
   bool Good() const { return (error_type == 0); }
   bool Help() const { return (error_type == 1); }
   void PrintOptions(std::ostream &out) const;
   void PrintError(std::ostream &out) const;
   void PrintHelp(std::ostream &out) const;
   void PrintUsage(std::ostream &out) const;
};
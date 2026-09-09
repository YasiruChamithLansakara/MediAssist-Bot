// ESLint 9 flat config.
//
// `react.configs.recommended` and `jsxA11y.configs.recommended` are legacy
// eslintrc objects: they declare `plugins` as an array of strings, which flat
// config rejects outright ("A config object has a 'plugins' key defined as an
// array of strings"). Lint has therefore been failing to start at all. The
// flat variants (`flat.recommended`) are used instead, with a fallback so the
// config keeps working if a plugin version does not ship one.

import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import react from "eslint-plugin-react";
import jsxA11y from "eslint-plugin-jsx-a11y";
import { defineConfig, globalIgnores } from "eslint/config";

const reactFlat = react.configs.flat?.recommended ?? {};
const a11yFlat = jsxA11y.flatConfigs?.recommended ?? {};

export default defineConfig([
  globalIgnores(["dist", "node_modules"]),

  {
    files: ["**/*.{js,jsx}"],

    extends: [
      js.configs.recommended,
      reactFlat,
      a11yFlat,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],

    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: "latest",
        ecmaFeatures: { jsx: true },
        sourceType: "module",
      },
    },

    settings: {
      react: { version: "detect" },
    },

    rules: {
      "no-unused-vars": ["error", { varsIgnorePattern: "^[A-Z_]" }],

      // The new JSX transform means React need not be in scope.
      "react/react-in-jsx-scope": "off",
      "react/prop-types": "off",

      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",

      "no-duplicate-imports": "error",
    },
  },
]);

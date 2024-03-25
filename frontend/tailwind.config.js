/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Red Hat Text'],
        display: ['Red Hat Display']
      },
      screens: {
        '3xl': '1792px'
      },
      colors: {
        light: {
          menu: 'rgba(240, 240, 235, 1)',
          content: 'rgba(255, 255, 255, 1)',
          distinct: 'rgba(246, 248, 250, 1)',
          text: 'rgba(0, 0, 0, 0.85)',
          'distinct-text': 'rgba(90, 90, 90, 0.9)',
          'disabled-text': 'rgba(180, 180, 180, 0.9)',
          'link-text': 'rgba(0, 92, 184, 0.9)',
          'link-text-hover': 'rgba(0, 59, 119, 0.9)'
        },

        dark: {
          menu: 'rgba(22, 27, 34, 1)',
          content: 'rgba(6, 8, 15, 1)',
          distinct: 'rgba(19, 19, 22, 1)',
          text: 'rgba(255, 255, 255, 0.8)',
          'distinct-text': 'rgba(150, 150, 150, 0.85)',
          'disabled-text': 'rgba(100, 100, 100, 0.9)',
          'link-text': 'rgba(86, 167, 252, 0.9)',
          'link-text-hover': 'rgba(134, 192, 253, 0.9)'
        }
      }
    }
  },
  plugins: []
}

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{html,ts}",
    "./node_modules/flowbite/**/*.js" // add this line
  ],
  theme: {
    extend: {
      width: {
        '100': '25rem',
        '200': '30rem',
      },
      height: {
        '100':'25rem',
        '200':'30rem',
        '300':'35rem',
        '400':'40rem',
      }
    },
  },
  plugins: [
    require('flowbite/plugin')
  ],
}

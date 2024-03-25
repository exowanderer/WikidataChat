<template>
  <main class="w-screen h-screen bg-light-content dark:bg-dark-content">

    <Settings v-if="showSettings" @close="showSettings = false" />

    <Icon
      class="text-4xl absolute right-2 top-2 cursor-pointer"
      icon="fluent:settings-24-filled"
      @click="showSettings = true"
    />

    <div class="py-8 space-y-8">
      <h1
        class="flex justify-center space-x-3 text-4xl md:text-5xl text-light-text dark:text-dark-text font-display"
      >
        <Icon icon="ooui:logo-wikimedia" />
        <p class="-mt-[0.025em] md:mt-0">ChatGSWiki</p>
      </h1>

      <div v-if="!apiSecret()"></div>
      <div v-else>
        <div class="flex justify-center w-full mb-4">
          <div class="flex-col w-[90%] md:w-4/5 lg:w-2/3">
            <div
              class="relative flex text-2xl rounded-lg bg-light-menu dark:bg-dark-menu elem-shadow-sm"
            >
              <input
                v-model="inputText"
                type="text"
                class="w-full pl-4 bg-transparent rounded-lg h-11 placeholder:text-light-distinct-text dark:placeholder:text-dark-distinct-text text-light-text dark:text-dark-text"
                :placeholder="$t('chat-prompt')"
                autocomplete="off"
                @keyup.enter="inputText.length > 0 ? search() : {}"
                @focus="inputFocused = true"
                @blur="inputFocused = false"
              />
              <Icon
                class="absolute -translate-y-1/2 right-3 top-1/2"
                :class="{
                  'text-light-text dark:text-dark-text ': inputFocused && inputText.length === 0,
                  'text-light-text dark:text-dark-text hover:text-light-distinct-text dark:hover:text-dark-distinct-text hover:cursor-pointer':
                    inputFocused && inputText.length > 0,
                  'text-light-distinct-text dark:text-dark-distinct-text': !inputFocused
                }"
                icon="fluent:send-24-filled"
                size="2em"
                @click="inputText.length > 0 ? search() : {}"
              />
            </div>
            <p class="pt-2 pl-4 text-sm text-light-distinct-text dark:text-dark-distinct-text">
              {{ $t('check-linked-wiki-pages-notice') }}
            </p>
          </div>
        </div>
        <div v-if="error && error.length" class="flex text-xl font-bold justify-center w-full">
          {{ error }}
        </div>
        <div v-if="response" class="flex justify-center w-full">
          <div class="flex-col w-[90%] md:w-4/5 lg:w-2/3 space-y-7">
            <!-- <FieldQuestion :text="inputText" /> -->
            <FieldAnswer :response="response" :isLoading="false" />
          </div>
        </div>
        <div v-else-if="displayResponse && !response" class="flex justify-center w-full">
          <div class="flex-col w-[90%] md:w-4/5 lg:w-2/3 space-y-5">
            <FieldAnswer :isLoading="true" />
          </div>
        </div>
      </div>
    </div>
  </main>
</template>

<script setup lang="ts">
import { Icon } from '@iconify/vue'
import { ref } from 'vue'
import FieldAnswer from '../components/field/FieldAnswer.vue'
// import FieldQuestion from '../components/field/FieldQuestion.vue'
import type { ResponseObject } from '../types/response-object.d.ts'
import Settings from '../components/Settings.vue'

const inputText = ref('')
const response = ref<ResponseObject>()
const error = ref<string>()
const displayResponse = ref(false)
const inputFocused = ref(false)
const showSettings = ref(!apiSecret())

// response.value = {
//   answer:
//     'Confidential information should not be used or disclosed except as necessary for performance under the agreement.',
//   sources: [
//     {
//       id: 1,
//       src: 'https://example.com/document1',
//       content:
//         'Parties agree that Confidential Information shall not be used or disclosed except as necessary for performance hereunder.',
//       score: 4.975016137070867
//     },
//     {
//       id: 2,
//       src: 'https://example.com/document12',
//       content: 'Any waiver of compliance with any provision of this Agreement must be in writing.',
//       score: 2.1152231308302407
//     },
//     {
//       id: 3,
//       src: 'https://example.com/document19',
//       content:
//         'This Agreement shall be governed by and construed in accordance with the laws of the State of New York.',
//       score: 1.8400423245643809
//     }
//   ]
// }

function apiSecret() {
  const apiSecret = sessionStorage.getItem('api-secret')
  if (!apiSecret || !apiSecret.length) {
    return null
  }
  return apiSecret
}

async function search() {
  response.value = undefined
  error.value = undefined

  displayResponse.value = true

  const secret = apiSecret()

  if (!secret) throw 'API secret not set.'

  try {
    const fetchResult = await fetch(`/api?query=${inputText.value}`, {
      headers: { 'x-api-secret': secret }
    })
    const jsonResponse = await fetchResult.json()
    response.value = jsonResponse
  } catch (e) {
    displayResponse.value = false
    console.log(e)
    error.value = 'Sorry. Failed to retrieve response.'
  }
}
</script>

<template>
  <div class="absolute z-10 bg-light-content dark:bg-dark-content w-screen h-screen">
    <div class="flex flex-row-reverse">
      <Icon
        class="text-4xl m-2 "
        :class="{
          'text-light-disabled-text dark:text-dark-disabled-text':
            !apiSecret(),
          'cursor-pointer text-light-text dark:text-dark-text hover:text-light-distinct-text dark:hover:text-dark-distinct-text':
            apiSecret()
        }"
        icon="fluent:dismiss-24-filled"
        @click="apiSecret() && $emit('close')"
      />
    </div>

    <div class="px-24 py-4 pb-24 max-w-3xl">
      <div class="text-4xl flex">
        <Icon class="mr-3 mb-4" icon="fluent:settings-24-filled" />
        Settings
      </div>
      <div>
      </div>
      <div class="relative flex text-2xl rounded-lg bg-light-menu dark:bg-dark-menu elem-shadow-sm">
        <Icon
          class="absolute -translate-y-1/2 left-3 top-1/2"
          :class="{
            'text-light-text dark:text-dark-text hover:text-light-distinct-text dark:hover:text-dark-distinct-text hover:cursor-pointer':
              inputText.length > 0
          }"
          icon="fluent:lock-closed-24-filled"
          size="2em"
        />

        <input
          v-model="inputText"
          type="password"
          class="w-full pl-12 bg-transparent rounded-lg h-11 placeholder:text-light-distinct-text dark:placeholder:text-dark-distinct-text text-light-text dark:text-dark-text"
          :placeholder="$t('enter-api-secret')"
          autocomplete="off"
          @input="storeSecret()"
          @keyup.enter="storeSecret()"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Icon } from '@iconify/vue'
import { ref } from 'vue'

const inputText = ref(apiSecret() || '')

function storeSecret() {
  sessionStorage.setItem('api-secret', inputText.value)
}

function apiSecret() {
  const apiSecret = sessionStorage.getItem('api-secret')
  if (!apiSecret || !apiSecret.length) {
    return null
  }
  return apiSecret
}
</script>

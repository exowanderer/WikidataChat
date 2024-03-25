<template>
  <div class="flex space-x-6 text-light-text dark:text-dark-text">
    <div class="p-1 text-3xl rounded-full bg-light-menu dark:bg-dark-menu h-min elem-shadow-sm">
      <Icon icon="fluent:person-24-filled" />
    </div>
    <div>
      <div class="text-lg">{{ text }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { Icon } from '@iconify/vue'

defineProps<{
  text: string
}>()
</script>

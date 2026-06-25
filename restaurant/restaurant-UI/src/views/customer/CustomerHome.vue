<template>
  <div class="customer-home">
    <!-- 轮播图 -->
    <el-carousel height="300px" class="banner-carousel" v-if="banners.length > 0">
      <el-carousel-item v-for="banner in banners" :key="banner.id">
        <img :src="banner.image" :alt="banner.title" class="banner-image" />
      </el-carousel-item>
    </el-carousel>

    <!-- 门店列表 -->
    <h2 class="section-title">附近门店</h2>
    <el-row :gutter="20">
      <el-col :span="8" v-for="store in stores" :key="store.id">
        <el-card shadow="hover" class="store-card" @click="goToStore(store.id)">
          <div class="store-cover">
            <img :src="store.coverImage || defaultCover" :alt="store.name" />
          </div>
          <div class="store-info">
            <h3 class="store-name">{{ store.name }}</h3>
            <p class="store-address">{{ store.address }}</p>
            <p class="store-hours">营业时间: {{ store.businessHours }}</p>
            <p class="store-price">人均: <span class="price-value">¥{{ store.avgPrice }}</span></p>
          </div>
        </el-card>
      </el-col>
    </el-row>

    <el-empty v-if="stores.length === 0 && !loading" description="暂无门店" />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { getBanners, getStores } from '@/api'

const router = useRouter()
const banners = ref([])
const stores = ref([])
const loading = ref(false)

const defaultCover = 'https://via.placeholder.com/400x200?text=火锅'

const loadBanners = async () => {
  try {
    const res = await getBanners({ page: 1, pageSize: 10 })
    banners.value = (res.data?.records || res.data?.list || res.data || []).filter(b => b.status === 1)
  } catch (e) {
    // ignore
  }
}

const loadStores = async () => {
  loading.value = true
  try {
    const res = await getStores({ page: 1, pageSize: 20 })
    stores.value = (res.data?.records || res.data?.list || res.data || []).filter(s => s.status === 1)
  } catch (e) {
    ElMessage.error('加载门店列表失败')
  } finally {
    loading.value = false
  }
}

const goToStore = (storeId) => {
  router.push(`/c/stores/${storeId}`)
}

onMounted(() => {
  loadBanners()
  loadStores()
})
</script>

<style scoped>
.customer-home {
  padding-bottom: 20px;
}

.banner-carousel {
  border-radius: 12px;
  overflow: hidden;
  margin-bottom: 30px;
}

.banner-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.section-title {
  font-size: 22px;
  margin-bottom: 20px;
  color: #333;
}

.store-card {
  margin-bottom: 20px;
  cursor: pointer;
  transition: transform 0.2s;
}

.store-card:hover {
  transform: translateY(-4px);
}

.store-card :deep(.el-card__body) {
  padding: 0;
}

.store-cover {
  width: 100%;
  height: 200px;
  overflow: hidden;
}

.store-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.store-info {
  padding: 16px;
}

.store-name {
  font-size: 18px;
  margin: 0 0 8px 0;
  color: #333;
}

.store-address,
.store-hours {
  font-size: 13px;
  color: #999;
  margin: 4px 0;
}

.store-price {
  font-size: 14px;
  color: #666;
  margin: 8px 0 0 0;
}

.price-value {
  color: #ff6b35;
  font-weight: bold;
  font-size: 16px;
}
</style>

package com.nigam.ancache.core;

import java.sql.Timestamp;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

import com.nigam.ancache.exception.CacheCapacityExceededException;
import com.nigam.ancache.model.CacheElement;

public class ANCacheManager<K, V> implements ANCache<K, V> {

	private static final long _1000000000L = 1000000000l;
	private final Integer ttlSeconds;
	private final Long ttlNanos;
	private final Integer capacity;

	private Map<K, V> cacheMap;
	private final Map<K, Timestamp> timestampMap;

	/**
	 * ß
	 * @param ttlSeconds
	 * @param capacity
	 */
	public ANCacheManager(Integer ttlSeconds, Integer capacity) {
		super();
		this.ttlSeconds = ttlSeconds;
		this.capacity = capacity;
		this.ttlNanos = this.ttlSeconds * ANCacheManager._1000000000L;
		this.cacheMap = new ConcurrentHashMap<K, V>();
		this.timestampMap = new ConcurrentHashMap<K, Timestamp>();
	}

	/**
	 * 
	 */
	public CacheElement<K, V> add(K k, V v) {
		if (cacheMap.size() < this.capacity && timestampMap.size() < this.capacity) {
			this.cacheMap.put(k, v);
			this.timestampMap.put(k, new Timestamp(System.currentTimeMillis()));
			return new CacheElement<K, V>(k, v);
		} else if (cacheMap.size() >= this.capacity && timestampMap.size() >= this.capacity) {
			removeOldEntries();
			this.cacheMap.put(k, v);
			this.timestampMap.put(k, new Timestamp(System.currentTimeMillis()));
			return new CacheElement<K, V>(k, v);
		} else {
			throw new CacheCapacityExceededException(
					" Cache already reached to its capacity of " + this.capacity + " elements.");
		}
	}

	/**
	 * 
	 */
	public CacheElement<K, V> update(V v) throws NoSuchElementException {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * 
	 */
	public CacheElement<K, V> addOrUpdate(K k, V v) {
		// TODO Auto-generated method stub
		return null;
	}

	/**
	 * 
	 */
	public Boolean evictByKey(K k) {
		if (this.cacheMap.containsKey(k) && this.timestampMap.containsKey(k)) {
			this.cacheMap.remove(k);
			this.timestampMap.remove(k);
			return Boolean.TRUE;
		} else {
			return Boolean.FALSE;
		}
	}

	/**
	 * 
	 */
	public Boolean evictByValue(V v) {

		Set<K> keySet = this.cacheMap.entrySet().stream()
				.filter(entry -> entry.getValue().equals(v))
				.map(entry -> entry.getKey()).collect(Collectors.toSet());
		
		return this.cacheMap.keySet().removeAll(keySet);
	}

	/**
	 * 
	 */
	public Set<CacheElement<K, V>> searchByValue(V v) {

		return this.cacheMap.entrySet().stream()
				.filter(entry -> entry.getValue().equals(v))
				.map(entry -> new CacheElement<>(entry.getKey(), entry.getValue()))
				.collect(Collectors.toSet());
	}

	/**
	 * 
	 */
	public CacheElement<K, V> get(K k) {
		CacheElement<K, V> cacheElement = null;
		if (this.cacheMap.containsKey(k) && this.timestampMap.containsKey(k)) {
			this.timestampMap.put(k, new Timestamp(System.currentTimeMillis()));
			cacheElement = new CacheElement<K, V>(k, this.cacheMap.get(k));
			return cacheElement;
		} else {
			throw new NoSuchElementException("No element found with key " + k.toString());
		}
	}
	
	private void removeOldEntries() {
		final Set<K> keySet =  identifyOldEntries();
		if(keySet != null) {
			this.timestampMap.keySet().removeAll(keySet);
			this.cacheMap.keySet().removeAll(keySet);
		} else {
			
		}
	}
	
	private Set<K> identifyOldEntries() {
		final Timestamp ts = new Timestamp(System.nanoTime() - this.ttlNanos);
		return this.timestampMap.entrySet().stream()
				.filter(entry -> entry.getValue().after(ts))
				.map(entry -> entry.getKey()).collect(Collectors.toSet());
		
	
				
		
	}

}

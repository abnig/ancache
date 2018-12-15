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

	private final Integer ttlSeconds;
	private final Integer capacity;

	private Map<K, V> cacheMap;
	private final Map<K, Timestamp> timestampMap;

	public ANCacheManager(Integer ttlSeconds, Integer capacity) {
		super();
		this.ttlSeconds = ttlSeconds;
		this.capacity = capacity;
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
		return null;
	}

	/**
	 * 
	 */
	public Set<CacheElement<K, V>> searchByValue(V v) {
		
		return this.cacheMap.entrySet().stream() 
		.filter(entry -> entry.getValue().equals(v)) // filter entries with same value 'v' 
		.map(entry -> new CacheElement<>(entry.getKey(), entry.getValue())) // map key and value to CacheElement 
		.collect(Collectors.toSet()); // collect as a Set 
		
	}
	/**
	 * 
	 */
	public CacheElement<K, V> searchByKey(K k) {
		CacheElement<K, V> cacheElement = null;
		if (this.cacheMap.containsKey(k) && this.timestampMap.containsKey(k)) {
			this.timestampMap.put(k, new Timestamp(System.currentTimeMillis()));
			cacheElement = new CacheElement<K, V>(k, this.cacheMap.get(k));
			return cacheElement;
		} else {
			throw new NoSuchElementException("No element found with key " + k.toString());
		}
	}

}

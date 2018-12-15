package com.nigam.ancache.model;

public final class CacheElement<K, V> {
	
	private final K k;
	private final V v;
	
	public CacheElement(K k, V v) {
		super();
		this.k = k;
		this.v = v;
	}
	
	public K getK() {
		return k;
	}
	
	public V getV() {
		return v;
	}
}
